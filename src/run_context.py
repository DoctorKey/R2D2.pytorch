# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

from datetime import datetime
from collections import defaultdict
import threading
import time
import logging
import os
#from tensorboardX import SummaryWriter

class RunContext:
    """Creates directories and files for the run"""

    def __init__(self, runner_file, run_idx):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        dataset = runner_file.split('/')[1]
        network = runner_file.split('/')[2]
        runner_name = os.path.basename(runner_file).split(".")[0]
        self.result_dir = "{root}/{dataset}/{network}/{runner_name}/{date:%Y-%m-%d_%H:%M:%S}/{run_idx}".format(
            root='results',
            dataset=dataset,
            network=network,
            runner_name=runner_name,
            date=datetime.now(),
            run_idx=run_idx
        )
        self.transient_dir = self.result_dir + "/transient"   
        os.makedirs(self.result_dir)
        os.makedirs(self.transient_dir)
        from torch.utils.tensorboard import SummaryWriter
        self.vis_log = SummaryWriter(self.result_dir + "/vis_log")
        self._init_log()

    def _init_log(self):
        LOG = logging.getLogger('main')
        FileHandler = logging.FileHandler(os.path.join(self.result_dir, 'log.txt'))
        LOG.addHandler(FileHandler)

