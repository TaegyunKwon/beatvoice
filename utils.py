import ntpath
import os


def maybe_make_dir(dir_name):
  if not os.path.isdir(dir_name):
    os.makedirs(dir_name)

def split_head_and_tail(file_path):
  head, tail = ntpath.split(file_path)
  return head, tail