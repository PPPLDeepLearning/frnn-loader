#!/usr/bin.env python
# -*- coding: UTF-8 -*-
#
# run with
# python -m unittest tests/test_machines.py
import unittest

from frnn_loader.backends.machine import MachineNSTX, MachineJET, MachineD3D


class TestMachines(unittest.TestCase):
    """Test routines for machines."""
    def test_machine_nstx(self):
        """Test whether we can instantiate NSTX machine"""
        my_machine = MachineNSTX()
        print(my_machine.current_threshold)

    def test_machine_jet(self):
        """Test whether we can instantiate JET machine"""
        my_machine = MachineJET()
        print(my_machine.current_threshold)

    def test_machine_d3d(self):
        """Test whether we can instantiate D3D machine"""
        my_machine = MachineD3D()
        print(my_machine.current_threshold)


if __name__ == "__main__":
    unittest.main()
