# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 20:31:06 2019

@author: aczd087
"""

class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)