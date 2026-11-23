"""
Created on 2017-4-25

@author: cheng.li
"""

import numpy as np

from alphamind.data.winsorize import NormalWinsorizer, winsorize_normal


class TestWinsorize:
    def setup_method(self):
        np.random.seed(10)
        self.x = np.random.randn(3000, 10)
        self.groups = np.random.randint(10, 30, size=3000)
        self.num_stds = 2

    def test_winsorize_normal(self):
        calc_winsorized = winsorize_normal(self.x, self.num_stds)
        std_values = self.x.std(axis=0, ddof=1)
        mean_value = self.x.mean(axis=0)
        lower_bound = mean_value - self.num_stds * std_values
        upper_bound = mean_value + self.num_stds * std_values
        for i in range(np.size(calc_winsorized, 1)):
            col_data = self.x[:, i]
            col_data[col_data > upper_bound[i]] = upper_bound[i]
            col_data[col_data < lower_bound[i]] = lower_bound[i]
            calculated_col = calc_winsorized[:, i]
            np.testing.assert_array_almost_equal(col_data, calculated_col)

    def test_winsorize_normal_with_interp(self):
        calc_winsorized = winsorize_normal(self.x, self.num_stds, method="interp")
        std_values = self.x.std(axis=0, ddof=1)
        mean_value = self.x.mean(axis=0)
        lower_bound = mean_value - self.num_stds * std_values
        upper_bound = mean_value + self.num_stds * std_values
        for i in range(np.size(calc_winsorized, 1)):
            col_data = self.x[:, i].copy()
            idx = col_data > upper_bound[i]
            u_values = col_data[idx]
            q_values = u_values.argsort().argsort()
            if len(q_values) > 0:
                col_data[idx] = upper_bound[i] + q_values / len(q_values) * 0.5 * std_values[i]
            idx = col_data < lower_bound[i]
            l_values = col_data[idx]
            q_values = (-l_values).argsort().argsort()
            if len(q_values) > 0:
                col_data[idx] = lower_bound[i] - q_values / len(q_values) * 0.5 * std_values[i]
            calculated_col = calc_winsorized[:, i]
            np.testing.assert_array_almost_equal(col_data, calculated_col)

    def test_winsorize_normal_with_group(self):
        cal_winsorized = winsorize_normal(self.x, self.num_stds, groups=self.groups)

        def impl(x):
            std_values = x.std(axis=0, ddof=1)
            mean_value = x.mean(axis=0)
            lower_bound = mean_value - self.num_stds * std_values
            upper_bound = mean_value + self.num_stds * std_values
            res = np.where(x > upper_bound, upper_bound, x)
            res = np.where(res < lower_bound, lower_bound, res)
            return res

        exp_winsorized = np.empty_like(self.x)
        for group in np.unique(self.groups):
            mask = self.groups == group
            exp_winsorized[mask] = impl(self.x[mask])
        np.testing.assert_array_almost_equal(cal_winsorized, exp_winsorized)

    def test_winsorize_normal_with_group_and_interp(self):
        cal_winsorized = winsorize_normal(
            self.x, self.num_stds, groups=self.groups, method="interp"
        )

        def impl(x):
            result = np.empty_like(x)
            for column in range(x.shape[1]):
                values = x[:, column]
                std = values.std(ddof=1)
                mean = values.mean()
                lower = mean - self.num_stds * std
                upper = mean + self.num_stds * std
                col_data = values.copy()
                idx = col_data > upper
                q_values = col_data[idx].argsort().argsort()
                if len(q_values) > 0:
                    col_data[idx] = upper + q_values / len(q_values) * 0.5 * std
                idx = col_data < lower
                q_values = (-col_data[idx]).argsort().argsort()
                if len(q_values) > 0:
                    col_data[idx] = lower - q_values / len(q_values) * 0.5 * std
                result[:, column] = col_data
            return result

        exp_winsorized = np.empty_like(self.x)
        for group in np.unique(self.groups):
            mask = self.groups == group
            exp_winsorized[mask] = impl(self.x[mask])
        np.testing.assert_array_almost_equal(cal_winsorized, exp_winsorized)

    def test_normal_winsorizer(self):
        s = NormalWinsorizer(num_stds=self.num_stds)
        s.fit(self.x)
        calc_winsorized1 = s.transform(self.x)
        calc_winsorized2 = s(self.x)
        std_values = self.x.std(axis=0, ddof=1)
        mean_value = self.x.mean(axis=0)
        lower_bound = mean_value - self.num_stds * std_values
        upper_bound = mean_value + self.num_stds * std_values
        for i in range(np.size(calc_winsorized1, 1)):
            col_data = self.x[:, i]
            col_data[col_data > upper_bound[i]] = upper_bound[i]
            col_data[col_data < lower_bound[i]] = lower_bound[i]
            calculated_col = calc_winsorized1[:, i]
            np.testing.assert_array_almost_equal(col_data, calculated_col)
            calculated_col = calc_winsorized2[:, i]
            np.testing.assert_array_almost_equal(col_data, calculated_col)
