# !/usr/bin/python

import os
for root, dirs, files in os.walk("C:/Users/selin/PycharmProjects/SampleProject/1150haber/raw_texts", topdown=False):
    for name in files:
        print(os.path.join(root, name))
    for name in dirs:
        print(os.path.join(root, name))
