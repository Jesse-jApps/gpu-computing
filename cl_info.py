import pyopencl as cl

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCL and graphic card specific data')
    parser.add_argument('-d', '--devices',  help='platform and device informations', action='store_true')
    args = parser.parse_args()

    if args.devices:
        for i, platform in enumerate(cl.get_platforms()):
            print("Platform %d" % i)
            print("- Name: %s" % platform.name)
            print("- OpenCL version: %s" % platform.version)
            for l, device in enumerate(platform.get_devices()):
                print("-- Device %d" % l)
                print("-- Name: %s" % device.name)
                print("-- Max comupte units: %s" % device.max_compute_units)
                print("-- Max workgroups sizes: %s" % device.max_work_item_sizes)
                print("-- DP support: %s " % ("Yes" if "cl_khr_fp64" in device.extensions else "No"))


    else:
        parser.print_help()

