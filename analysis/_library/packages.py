#!/bin/python
# -*- coding: utf-8 -*-

__author__      = 'Roy Gardner'
__copyright__   = 'Copyright 2023-2024, Roy and Sally Gardner'

import angular_distance as ad

import csv
from datetime import datetime, timedelta
from decimal import *

from IPython.core.display import HTML
from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets import Layout, HBox, VBox
import ipywidgets as widgets
from IPython.display import Image, display, clear_output
from IPython.display import Javascript

import networkx as nx

import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import random

import scipy as sp
from scipy import stats
from scipy.spatial.distance import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import string

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import time

