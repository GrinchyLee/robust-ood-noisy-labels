#!/bin/bash
python main.py models/tcl/configs/tin/seed1/tinyimagenet_sym_0.2_r18.yml
python main.py models/tcl/configs/tin/seed1/tinyimagenet_clean_r18.yml
python main.py models/tcl/configs/tin/seed1/tinyimagenet_sym_0.5_r18.yml
python main.py models/tcl/configs/tin/seed1/tinyimagenet_asym_0.4_r18.yml

python main.py models/tcl/configs/tin/seed2/tinyimagenet_sym_0.2_r18.yml
python main.py models/tcl/configs/tin/seed2/tinyimagenet_clean_r18.yml
python main.py models/tcl/configs/tin/seed2/tinyimagenet_sym_0.5_r18.yml
python main.py models/tcl/configs/tin/seed2/tinyimagenet_asym_0.4_r18.yml