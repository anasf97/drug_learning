import os
import pytest
import drug_learning.two_dimensions.Input.fingerprints as fp

DIR = os.path.dirname(__file__)
MOLECULE = os.path.join(DIR, "data/test_molecule.sdf")

def test_fp_fit(molecule = MOLECULE):
    fingerprint = fp.bc.Fingerprint()
    fingerprint.fit(molecule)
    assert len(fingerprint.structures) == 1

@pytest.mark.parametrize( "fp_class, shape",
                        [
                            (fp.MorganFP, (1, 2048)),
                            (fp.MACCS_FP, (1, 167)),
                            (fp.RDkitFP, (1, 2048)),
                            (fp.MordredFP, (1, 1613)) # mirar error
                        ]
                            )
def test_fingerprint_transform(fp_class, shape, molecule = MOLECULE):
    fingerprint = fp_class()
    fingerprint.fit(molecule)
    fingerprint.transform()
    assert fp.np.shape(fingerprint.features) == shape
