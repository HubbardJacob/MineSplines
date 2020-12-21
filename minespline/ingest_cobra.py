import sys

from minespline.config import C

# %%
import rasterio as rio
import os
import xmltodict
import easydict
import geojson
# %%
import logging

L = logging.getLogger(__name__)

# %%

CHANNELS = [
    'Infrared_Primary',
    'Infrared_Secondary',
    'Red_Secondary',
    'Orange_Primary',
    'Orange_Secondary',
    # 'Green_Primary',
    'Green_Secondary',
    'Blue_Secondary',
    'Violet_Secondary',
]

# %%

# %%

# Format is "{basedir}/{mission_id}_{channel}.nitf"
TEST_PREFIX = '/media/femianjc/USB30FD/Curvilinear GFI package/curvilinear xml and imagery/2008.04.14.15.29.29.Spot_00025.COL_001_00020'


# %%

def merge_nitf_files(prefix):
    """

    :param prefix:  all bands are formatted as {prefix}_{channelname}.nitf
    :return: path to output file {prefix}_stack.tif
    """
    # %%
    files = ['_'.join([prefix, channel + '.nitf']) for channel in CHANNELS]

    # %%
    for file in files:
        if not os.path.isfile(file):
            raise IOError(f'Cannot access file "{file}"')

    # %%
    # Read metadata of first file
    with rio.open(files[0]) as src0:
        meta = src0.meta
    # %%

    # Update meta to reflect the number of layers
    meta.update(count=len(files), driver='GTiff', compress='LZW', predictor=2)

    # %%

    # Read each layer and write it to stack
    with rio.open(f'{prefix}_stack.tif', 'w', **meta) as dst:
        for id, layer in enumerate(files, start=1):
            with rio.open(layer) as src1:
                dst.write_band(id, src1.read(1))
    # %%


def xml_to_geojson(filename, outfile=None):
    """
    Save the XML from 'filename' as a GEOJSON file.

    :param filename:
    :param outfile:
    :return:
    """

    if outfile is None:
        outfile=filename.replace('.xml', '.geojson')

    if isinstance(filename, (str, bytes)):
        with open(filename, 'rb') as f:
            xml = xmltodict.parse(f, force_list='THREAT')
    else:
        xml = xmltodict.parse(filename, force_list='THREAT')

    xml = easydict.EasyDict(xml)
    threats = xml.COBRA_EA_MESSAGE.PMA_REPORT.MISSION.THREAT

    features = []
    for threat in threats:
        geometry = geojson.Point((float(threat.POINT.POSITION.LONGITUDE),
                                  float(threat.POINT.POSITION.LATITUDE),
                                  float(threat.POINT.ELEVATION['#text'])))
        properties = dict(
            ALGORITHM_INFO_algo_type=threat.ALGORITHM_INFO['@algo_type'],
            ALGORITHM_INFO_algo_version=threat.ALGORITHM_INFO['@algo_version'],
            UDA_REF=threat.UDA_REF,
            THREAT_CLASS=threat.THREAT_CLASS,
            THREAT_TYPE=threat.THREAT_TYPE,
            FIELD=threat.FIELD,
            OBSERVATION_DTG=threat.OBSERVATION_DTG,
            PLACEMENT_type=threat.PLACEMENT['@type'],
            PLACEMENT_BURIAL_type=threat.PLACEMENT.BURIAL['@type'],
            POINT_ELEVATION_units=threat.POINT.ELEVATION['@units'],
            POINT_POSITION_LATITUDE=threat.POINT.POSITION.LATITUDE,
            POINT_POSITION_LONGITUDE=threat.POINT.POSITION.LONGITUDE,
            POINT_POSITION_ELLIPSE_SEMI_MAJOR=threat.POINT.POSITION.ELLIPSE.SEMI_MAJOR['#text'],
            POINT_POSITION_ELLIPSE_SEMI_MAJOR_units=threat.POINT.POSITION.ELLIPSE.SEMI_MAJOR['@units'],
            POINT_POSITION_ELLIPSE_SEMI_MINOR=threat.POINT.POSITION.ELLIPSE.SEMI_MINOR['#text'],
            POINT_POSITION_ELLIPSE_SEMI_MINOR_units=threat.POINT.POSITION.ELLIPSE.SEMI_MINOR['@units'],
            POINT_POSITION_ELLIPSE_ORIENTATION=threat.POINT.POSITION.ELLIPSE.ORIENTATION,
            REMARKS=threat.REMARKS,
        )

        feature = geojson.Feature(geometry=geometry, properties=properties)
        features.append(feature)

    features = geojson.FeatureCollection(features)

    if isinstance(outfile, (str, bytes)):
        outfile = open(outfile, 'w')

    geojson.dump(features, outfile)

    return outfile

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('file', help="A file (.xml|.nitf) to convert.")
    p.add_argument('--output', '-o', type=argparse.FileType('w'), default="-",
                   help="Output file. It is a .geojson file if the input is XML, or "
                        "a geotiff (.tif) otherwise. Use '-' to"
                        "save to standard output."
                        "\n"
                        "For .nitf files, the user can either pass in a single channel,"
                        "in which case the '<channel_name>.nitf' portion of the file name"
                        "will be removes to determine the prifex, and we will look for other"
                        "channels by replacing <channel_name> in order to create a merged"
                        "geotiff image, "
                        "\n"
                        f"The order of channels in our geotif is {','.join(CHANNELS)}.")
    args = p.parse_args()

    if args.file.endwith('.xml'):
        outfile = xml_to_geojson(args.file, outfile=args.output)

    else:
        if args.file.endwith('.nitf'):
            # Extract the prefix if the pass in a single channel
            for c in CHANNELS:
                if c in args.file:
                    args.file = args.file.replace(c + '.nitf', '')

        outfile = merge_nitf_files(args.file)
if __name__ == '__main__':
    main()