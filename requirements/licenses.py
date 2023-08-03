import glob
from os.path import join, dirname
from pathlib import Path

import pkg_resources


def get_pkg_license(pkg):
    if pkg is not None:
        try:
            lines = pkg.get_metadata_lines('METADATA')
        except:
            lines = pkg.get_metadata_lines('PKG-INFO')

        for line in lines:
            if line.startswith('License:'):
                return line[9:]

    return '(Licence not found)'


hardcoded = {
    'requests-auth-aws-sigv4': 'Apache 2.0'
}


def print_licenses():
    for path in glob.glob(join(dirname(__file__), '*.txt')):
        with open(path) as f:
            label = Path(path).stem

            print("-" * 80)
            print(label)
            print("-" * 80)

            libraries = f.read().splitlines()

            for library in libraries:
                library = library.split(' ')[0]

                if library in hardcoded:
                    license = hardcoded[library]
                else:
                    pkg = pkg_resources.working_set.by_key.get(library)
                    license = get_pkg_license(pkg)

                print('{0:25} | {1:20}'.format(library, license))


if __name__ == "__main__":
    print_licenses()
