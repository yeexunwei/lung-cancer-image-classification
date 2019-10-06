#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 00:05:18 2019

@author: hsunwei
"""
import os


INPUT_FOLDER = '../data/SPIE-AAPM/SPIE-AAPM Lung CT Challenge/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()