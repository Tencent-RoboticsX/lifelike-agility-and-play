import os


def get_data_path():
    resdir = os.path.join(os.path.dirname(__file__), "data")
    return resdir


def get_urdf_path():
    resdir = os.path.join(get_data_path(), "urdf")
    return resdir
