#!/usr/bin/env python3
# Copyright (c) 2019 SiFive Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This is a python script for generating RTL testbench Devicetree overlays from the Devicetree
for the RTL DUT.
"""

from targets.generic import set_boot_hart, set_stdout, set_entry, get_spi_flash, get_spi_region, get_rams, set_rams, get_ccache, get_ccache_region, set_ram, set_itim

#def attach_testrams(tree, overlay):

def generate_overlay(tree, overlay):
    """Generate the overlay"""
    bootrom = get_spi_flash(tree)

    model = tree.root().get_field("model")
    if model == "sifive,hifive1":
        offset = 0x400000
    elif model == "sifive,hifive1-revb":
        offset = 0x10000
    else:
        offset = 0x0

    if bootrom is not None:
        region = get_spi_region(bootrom)
        set_entry(overlay, bootrom, region, offset)

    set_boot_hart(tree, overlay)
    set_stdout(tree, overlay, 115200)

    if model == "sifive,hifive-unmatched-a00" or \
       model == "sifive,hifive-unleashed-a00":
        ram = itim = ccache = get_ccache(tree)
        region = get_ccache_region(ccache)
        set_rams(overlay, ram, itim, region)
    else:
        ram, itim = get_rams(tree)
        set_rams(overlay, ram, itim)
