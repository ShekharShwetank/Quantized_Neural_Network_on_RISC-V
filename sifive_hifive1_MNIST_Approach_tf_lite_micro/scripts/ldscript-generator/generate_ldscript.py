#!/usr/bin/env python3
# Copyright (c) 2020 SiFive Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate linker scripts from devicetree source files"""

import argparse
import sys

import jinja2
import pydevicetree

from memory_map import get_memories, get_ram_memories, get_load_map

TEMPLATES_PATH = "templates"

# Sets the threshold size of the ITIM at or above which the "ramrodata" layout
# places the text section into the ITIM
MAGIC_RAMRODATA_TEXT_THRESHOLD = 0x8000


def missingvalue(message):
    """
    Raise an UndefinedError
    This function is made available to the template so that it can report
    when required values are not present and cause template rendering to
    fail.
    """
    raise jinja2.UndefinedError(message)


def parse_arguments(argv):
    """Parse the arguments into a dictionary with argparse"""
    arg_parser = argparse.ArgumentParser(
        description="Generate linker scripts from Devicetrees")

    arg_parser.add_argument("-d", "--dts", required=True,
                            help="The path to the Devicetree for the target")
    arg_parser.add_argument("-o", "--output",
                            type=argparse.FileType('w'),
                            help="The path of the linker script file to output")
    group = arg_parser.add_mutually_exclusive_group()
    group.add_argument("--scratchpad", action="store_true",
                       help="Emits a linker script with the scratchpad layout")
    group.add_argument("--ramrodata", action="store_true",
                       help="Emits a linker script with the ramrodata layout")
    group.add_argument("--freertos", action="store_true",
                       help="Emits a linker script with specific layout for freertos")

    return arg_parser.parse_args(argv)


def get_template(parsed_args):
    """Initialize jinja2 and return the right template"""
    env = jinja2.Environment(
        loader=jinja2.PackageLoader(__name__, TEMPLATES_PATH),
        trim_blocks=True, lstrip_blocks=True,
    )
    # Make the missingvalue() function available in the template so that the
    # template fails to render if we don't provide the values it needs.
    env.globals["missingvalue"] = missingvalue

    if parsed_args.ramrodata:
        layout = "ramrodata"
    elif parsed_args.scratchpad:
        layout = "scratchpad"
    elif parsed_args.freertos:
        layout = "freertos"
    else:
        layout = "default"

    template = env.get_template("%s.lds" % layout)
    print("Generating linker script with %s layout" % layout, file=sys.stderr)

    return template


def print_memories(memories):
    """Report chosen memories to stdout"""
    print("Using layout:", file=sys.stderr)
    for _, memory in memories.items():
        end = memory["base"] + memory["length"] - 1
        print("\t%4s: 0x%08x-0x%08x (%s)" %
              (memory["name"], memory["base"], end, memory["path"]), file=sys.stderr)


def get_itim_length(memories):
    """Get the length of the itim, if it exists"""
    if "itim" in memories and "length" in memories["itim"]:
        return memories["itim"]["length"]
    return 0


def get_sorted_ram_memories(dts):
    """Get a sorted RAM list"""
    ram_memories = get_ram_memories(dts)
    sorted_ram_list = list(ram_memories.values())
    sorted_ram_list.sort(key=lambda m: m["name"])
    print("Consolidated RAM memories:", file=sys.stderr)
    for memory in sorted_ram_list:
        print("\t%4s: 0x%08x-0x%08x" %
              (memory["name"], memory["base"], memory["length"]), file=sys.stderr)
    return sorted_ram_list


def get_rnmis(harts):
    """
    Generate RNMI interrupt and exception handler sections for all cores.
    Cores may share same interrupt and exception handlers, so only the
    unique handler sections (i.e. not same address) are generated.
    """
    rnmis = []
    rnmi_irqs = set()
    rnmi_excps = set()
    rnmi_id = 0

    for hart in harts:
        rnmi_irq_addr = hart.get_fields("sifive,nmi-interrupt-vector")
        rnmi_excp_addr = hart.get_fields("sifive,nmi-exception-vector")

        if rnmi_irq_addr is not None and rnmi_excp_addr is not None:
            _rnmi_irq_addr = rnmi_irq_addr[0] << 32 | rnmi_irq_addr[1]
            _rnmi_excp_addr = rnmi_excp_addr[0] << 32 | rnmi_excp_addr[1]

            # Skip interupt and exception handler section
            # if we've already generated.
            if (_rnmi_irq_addr not in rnmi_irqs and
                    _rnmi_excp_addr not in rnmi_excps):

                rnmi = {
                    "id": rnmi_id,
                    "irq": {
                        "base_hex": "0x%x" % _rnmi_irq_addr,
                        "length_hex": "0x%x" % 4096,
                    },
                    "excp": {
                        "base_hex": "0x%x" % _rnmi_excp_addr,
                        "length_hex": "0x%x" % 4096,
                    }
                }
                rnmis.append(rnmi)
                rnmi_id += 1

                rnmi_irqs.add(_rnmi_irq_addr)
                rnmi_excps.add(_rnmi_excp_addr)
    return rnmis


def main(argv):
    """Parse arguments, extract data, and render the linker script to file"""
    # pylint: disable=too-many-locals

    parsed_args = parse_arguments(argv)

    template = get_template(parsed_args)

    dts = pydevicetree.Devicetree.parseFile(
        parsed_args.dts, followIncludes=True)

    memories = get_memories(dts)
    print_memories(memories)
    sorted_ram_memories = get_sorted_ram_memories(dts)

    ram, rom, itim, lim = get_load_map(memories, scratchpad=parsed_args.scratchpad)

    text_in_itim = False
    if parsed_args.ramrodata and get_itim_length(memories) >= MAGIC_RAMRODATA_TEXT_THRESHOLD:
        text_in_itim = True
        print(".text section included in ITIM", file=sys.stderr)
    elif parsed_args.ramrodata:
        print(".text section included in ROM", file=sys.stderr)

    harts = dts.get_by_path("/cpus").children
    chosenboothart = dts.chosen("metal,boothart")
    if chosenboothart:
        boot_hart = dts.get_by_reference(chosenboothart[0]).get_reg()[0][0]
    elif len(harts) > 1:
        boot_hart = 1
    else:
        boot_hart = 0

    if len(sorted_ram_memories) == 0:
        # If there are no rams to scrub, don't bother scrubbing them
        ecc_scrub = 0
    elif dts.chosen("metal,eccscrub"):
        # Otherwise default to scrubbing if metal,eccscrub = <1>;
        ecc_scrub = dts.chosen("metal,eccscrub")[0]
    else:
        ecc_scrub = 0

    # Pass sorted memories to the template generator so that the generated linker
    # script is reproducible.
    sorted_memories = list(memories.values())
    sorted_memories.sort(key=lambda m: m["name"])

    rnmis = get_rnmis(harts)

    values = {
        "memories": sorted_memories,
        "ram_memories": sorted_ram_memories,
        "default_stack_size": "0x400",
        "default_heap_size": "0x800",
        "num_harts": len(harts),
        "boot_hart": boot_hart,
        "chicken_bit": 1,
        "eccscrub_bit": ecc_scrub,
        "text_in_itim": text_in_itim,
        "rom": rom,
        "itim": itim,
        "lim": lim,
        "ram": ram,
        "rnmis": rnmis,
    }

    if parsed_args.output:
        parsed_args.output.write(template.render(values))
        parsed_args.output.close()
    else:
        print(template.render(values))


if __name__ == "__main__":
    main(sys.argv[1:])
