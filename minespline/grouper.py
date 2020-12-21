from minespline.config import C
import logging

L = logging.getLogger('grouper')


def groupmines(detections, groundtruth):
    pass


def main():
    from argparse import ArgumentParser
    p = ArgumentParser()

    p.add_argument('detections',
                   help=('Detections either as an image '
                         '(TIFF, JPG, NPZ, etc) or as a CSV file'))
    p.add_argument('--groundtruth', '-t', help='The locations of points that are truly mines')
    p.add_argument('--elevation', '-z', help='A digital elevation map', default=None)

    p.add_argument('--spacing', help='Mean spacing of mines', default=C.MODEL.SPACING)

    args = p.parse_args()

    L.error("This part of our tool is not yet implemented.")


if __name__ == '__main__':
    main()
