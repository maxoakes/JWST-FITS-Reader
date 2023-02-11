from astropy.io import fits
from Card import Card

class Metadata:
    # dataproduct_type
    # calib_level
    # obs_collection
    # obs_id
    # target_name
    # s_ra
    # s_dec
    # t_min
    # t_max
    # t_exptime
    # wavelength_region
    # filters
    # em_min
    # em_max
    # target_classification
    # obs_title
    # t_obs_release
    # instrument_name
    # proposal_pi
    # proposal_id
    # proposal_type
    # project
    # sequence_number
    # provenance_name
    # s_region
    # jpegURL
    # dataURL
    # dataRights
    # mtFlag
    # srcDen
    # intentType
    # obsid
    # objID
    row = {}
    fits_path = ""
    preview_path = ""

    def __init__(self, r):
        self.row = r

    def set_files(self, data, preview):
        self.fits_path = data
        self.preview_path = preview
        return (self.fits_path, self.preview_path)

    def __str__(self):
        return self.row['obs_id']
