"""
Created on 2018-2-9

@author: cheng.li
"""

from alphamind.data.engines.universe import Universe
from alphamind.model.composer import Composer, DataMeta
from alphamind.model.treemodel import XGBClassifier


class TestComposer:
    def _assert_composer_equal(self, lhs: Composer, rhs: Composer):

        assert type(lhs.alpha_model) is type(rhs.alpha_model)

        assert lhs.data_meta == rhs.data_meta

    def test_data_meta_persistence(self):
        freq = "5b"
        universe = Universe("zz800")
        batch = 4
        neutralized_risk = ["SIZE"]
        risk_model = "long"
        pre_process = ["standardize", "winsorize_normal"]
        post_process = ["standardize", "winsorize_normal"]
        warm_start = 2
        data_source = "postgresql://user:pwd@server/dummy"
        data_meta = DataMeta(
            freq=freq,
            universe=universe,
            batch=batch,
            neutralized_risk=neutralized_risk,
            risk_model=risk_model,
            pre_process=pre_process,
            post_process=post_process,
            warm_start=warm_start,
            data_source=data_source,
        )
        data_desc = data_meta.save()
        loaded_data = DataMeta.load(data_desc)

        assert data_meta.freq == loaded_data.freq

        assert data_meta.universe == loaded_data.universe

        assert data_meta.batch == loaded_data.batch

        assert data_meta.neutralized_risk == loaded_data.neutralized_risk

        assert data_meta.risk_model == loaded_data.risk_model

        assert data_meta.pre_process == loaded_data.pre_process

        assert data_meta.post_process == loaded_data.post_process

        assert data_meta.warm_start == loaded_data.warm_start

        assert data_meta.data_source == loaded_data.data_source

    def test_composer_persistence(self):
        freq = "5b"
        universe = Universe("zz800")
        batch = 4
        neutralized_risk = ["SIZE"]
        risk_model = "long"
        pre_process = ["standardize", "winsorize_normal"]
        post_process = ["standardize", "winsorize_normal"]
        warm_start = 2
        data_source = "postgresql://user:pwd@server/dummy"
        data_meta = DataMeta(
            freq=freq,
            universe=universe,
            batch=batch,
            neutralized_risk=neutralized_risk,
            risk_model=risk_model,
            pre_process=pre_process,
            post_process=post_process,
            warm_start=warm_start,
            data_source=data_source,
        )
        features = {"f1": "closePrice", "f2": "openPrice"}
        alpha_model = XGBClassifier(features=features)
        composer = Composer(alpha_model=alpha_model, data_meta=data_meta)
        comp_desc = composer.save()
        loaded_comp = Composer.load(comp_desc)
        self._assert_composer_equal(composer, loaded_comp)
