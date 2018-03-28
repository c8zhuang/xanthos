"""
Example script demonstrating how to call and run Xanthos.
"""

from xanthos.model import Xanthos


def run(ini):

    # instantiate model
    xth = Xanthos(ini)

    # run intended configuration
    xth.execute()

    return xth


if __name__ == "__main__":

    # full path to parameterized config file

    # Hargreaves PET - GWAM RUNOFF - MRTM ROUTING - HISTORIC
    # ini = '/users/ladmin/repos/github/xanthos/example/hargreaves_gwam_mrtm_hist.ini'

    # Hargreaves PET - GWAM RUNOFF - MRTM ROUTING - FUTURE
    # ini = '/users/ladmin/repos/github/xanthos/example/hargreaves_gwam_mrtm_futu.ini'

    ini = '/users/ladmin/repos/github/xanthos/example/hargreaves_abcd_mrtm_hist.ini'

    # run the model
    xth = run(ini)

    del xth
