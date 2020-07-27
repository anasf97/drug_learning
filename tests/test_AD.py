import os
import pytest
import numpy as np
import drug_learning.two_dimensions.main_fingerprints as mn

DIR = os.path.dirname(__file__)
MOLECULE1 = os.path.join(DIR, "data/test_ad_molecule1.sdf")
MOLECULE2 = os.path.join(DIR, "data/test_ad_molecule2.sdf")

def morgan_fp(input_sdf):
    morgan = mn.fp.MorganFP()
    morgan.fit(input_sdf)
    features = morgan.transform()
    return features

def test_AD_fit(features = morgan_fp(MOLECULE1)):
    AD = mn.ad.ApplicabilityDomain()
    AD.fit(features)
    assert np.all(AD.thresholds == np.array([0., 0.]))

@pytest.mark.parametrize( "features, other_features, result",
                            [
                            (morgan_fp(MOLECULE1), morgan_fp(MOLECULE1), [2,2] ),
                            (morgan_fp(MOLECULE1), morgan_fp(MOLECULE2), [0,0] ),
                            ]
)
def test_AD_predict(features, other_features, result):
    AD = mn.ad.ApplicabilityDomain()
    AD.fit(features)
    AD.predict(other_features)
    assert np.all(AD.n_insiders == np.array(result))
