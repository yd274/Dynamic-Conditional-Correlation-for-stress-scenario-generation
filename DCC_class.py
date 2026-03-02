"""
This file contain the class defining the Dynamic Conditional Correlation (DCC) model DCC is the multivariate extension
of GARCH model, which directly models the time-varying conditional correlation and thus, covariance, matrices of
asset returns
"""

import numpy as np
from arch.univariate import GARCH, ARCHInMean, Normal, StudentsT
from arch import arch_model
import scipy as sp
import pandas as pd
from statsmodels.base.model import GenericLikelihoodModel

class DCC(GenericLikelihoodModel):
    def __init__(self, y, mean=False, err_dist="Normal", vol_asymmetry=None, **kwargs):
        """
        :param y: input data
        :param mean: whether there is a drift term in the return series
        :param err_dist: whether to use Normal or StudentsT for the return innovation, Note that the conditional
        distribution of return will be heavy-tailed even with Normal innovations
        :param vol_asymmetry: a list of variables for which to include the leverage effect on volatility
        """
        super(DCC, self).__init__(y, **kwargs)
        self.y = y
        self._n_dim = y.shape[1]
        self.mu = np.mean(y, 0)
        self.sigma = np.std(y, 0)
        # Cholesky decompose the correlation matrix
        self.L0 = np.linalg.cholesky(np.corrcoef(y.T))
        self.mean = mean
        self.err_dist = err_dist
        self.vol_asymmetry = vol_asymmetry

    def volatility_model(self):
        """
        Method that fits a univariate GARCH model to the diagonal of covariance matrix, since they are simply the
        conditional variance of returns
        :return: volatility models for each asset
        """
        if self.vol_asymmetry:
            asym = 1
        else:
            asym = 0

        vol_model = []
        for i in range(self._n_dim):
            if isinstance(asym, list):
                if i in asym:
                    o = 1
                else:
                    o = 0
            else:
                if asym == 1:
                    o = 1
                else:
                    o = 0

           # Either fit a GJR_GARCH or GARCH models with normal or student t innovations, or GARCH-in-Mean model
            if self.mean and self.err_dist == "Normal":
                vol_model.append(ARCHInMean(self.y.iloc[:, i], volatility=GARCH(p=1, o=o, q=1), form="vol",
                                            distribution=Normal()).fit(disp=0))
            elif self.mean and self.err_dist == "StudentsT":
                vol_model.append(ARCHInMean(self.y.iloc[:, i], volatility=GARCH(p=1, o=o, q=1), form="vol",
                                            distribution=StudentsT()).fit(disp=0))
            else:
                vol_model.append(arch_model(self.y.iloc[:, i], mean="AR", p=1, o=o, q=1,
                                            dist=self.err_dist).fit(disp=0))

        return vol_model

    def conditional_cov(self, params):
        """
        The general formula for conditional covariance matrix is of the error correction form,
        Q[t] = A0 + A * (Q[t-1] - A0) + B * (stand_resid * stand_resid^T - A0), where Q[t] is the pseudo-correlation
        matrix at time t
        :param params: contains A the autoregressive term and B the error correction term
        :return: time series of conditional covariance and correlation matrices
        """
        T = self.y.shape[0]
        conditional_vol = np.zeros((T, self._n_dim))
        stand_resid = np.zeros((T, self._n_dim))
        resid = pd.DataFrame()
        resid.index = self.y.index
        mu = np.zeros((1, self._n_dim))
        vol_res = self.volatility_model()
        # Collect the conditional volatility and standardised residuals from univariate GARCH
        for i in range(self._n_dim):
            conditional_vol[:, i] = vol_res[i].conditional_volatility
            stand_resid[:, i] = vol_res[i].std_resid
            mu[:, i] = vol_res[i].params.Const
            temp = pd.DataFrame(vol_res[i].resid)
            temp.columns = [self.y.columns[i]]
            resid = resid.join(temp)

        # Initialize dynamic correlation such that the long-run mean is the unconditional correlation matrix
        A0 = self.L0 @ self.L0.T
        A = params[0]
        B = params[1]

        Q = np.zeros((T, self._n_dim, self._n_dim))
        Q[0] = A0
        # Initialize the conditional covariance and correlation matrices to the sample equivalents
        Corr = np.zeros((T, self._n_dim, self._n_dim))
        Corr[0] = np.corrcoef(self.y.T)
        Cov = np.zeros((T, self._n_dim, self._n_dim))
        Cov[0] = np.cov(self.y.T)

        for t in range(1, T):
            temp = np.reshape(np.array(stand_resid[t-1]), (self._n_dim, 1))
            # Use the squared residual matrix as an estimator for the conditional correlation
            temp = temp @ temp.T
            Q[t] = A0 + A * (Q[t - 1] - A0) + B * (temp - A0)

            # Ensure the diagonal is 1 since it is correlation matrix
            diag_Q = np.sqrt(np.diag(Q[t]))
            diag_Q = np.diag(diag_Q ** -1)
            Corr[t] = diag_Q @ Q[t] @ diag_Q

            # From correlation to covariance
            Cov[t] = np.diag(conditional_vol[t]) @ Corr[t] @ np.diag(conditional_vol[t])

        return {"Conditional correlation": Corr, "Conditional covariance": Cov, "Mean": mu, "Residual": resid,
                "Std residual": stand_resid, "Q": Q}

    @staticmethod
    def is_pos_def(cov):
        """
        To check if the covariance matrix is positive definite
        :param cov: Conditional covariance
        :return: True or False
        """
        return np.all(np.linalg.eigvals(cov) > 0)

    def loglike(self, params):
        """
        Function returns the log likelihood function of DCC
        """
        T = self.y.shape[0]
        Dcc_res = self.conditional_cov(params)
        Cov = Dcc_res["Conditional covariance"]
        demeaned_y = Dcc_res["Residual"]
        log_pdf = 0
        if self.err_dist == "StudentsT":
            nu = params[2]

        for t in range(T):
            if np.isnan(Cov[t]).any() or np.isinf(Cov[t]).any():
                log_pdf = -np.inf
            else:
                if self.is_pos_def(Cov[t]):
                    if self.err_dist == "StudentsT":
                        log_pdf += (sp.stats.multivariate_t(loc=None, shape=Cov[t] * ((nu - 2)/nu), df=nu).
                                    logpdf(demeaned_y.iloc[t, :]))
                    else:
                        log_pdf += sp.stats.multivariate_normal(mean=None, cov=Cov[t]).logpdf(demeaned_y.iloc[t, :])
                else:
                    log_pdf = -np.inf

        return log_pdf

    def forecast(self, params, step=1):
        A0 = self.L0 @ self.L0
        A = params[0]
        B = params[1]
        if self.err_dist == "StudentsT":
            nu = params[2]

        arch_forecast = []
        vol_forecast = np.zeros((step, self._n_dim))
        mean_forecast = np.zeros((step, self._n_dim))
        vol_res = self.volatility_model()

        # Get the conditional variance and mean forecast from GARCH
        for i in range(self._n_dim):
            arch_forecast.append(vol_res[i].forecast(horizon=step, reindex=False))
            vol_forecast[:, i] = np.sqrt(arch_forecast[i].residual_vairance)
            mean_forecast[:, i] = arch_forecast[i].mean

        temp_res = self.conditional_cov(params)
        last_values = temp_res["Std residual"][-1]
        Q0 = temp_res(params)["Q"]
        temp = np.reshape(np.array(last_values), (self._n_dim, 1))
        temp = temp @ temp.T
        Corr_forecast = np.zeros((step, self._n_dim, self._n_dim))
        Q = np.zeros((step, self._n_dim, self._n_dim))
        Cov_forecast = np.zeros((step, self._n_dim, self._n_dim))
        Q[0] = A0 + A * (Q0 - A0) + B * (temp - A0)
        diag_Q = np.sqrt(np.diag(Q[0]))
        diag_Q = np.diag(diag_Q ** -1)
        Corr_forecast[0] = diag_Q @ Q[0] @ diag_Q
        Cov_forecast[0] = np.diag(vol_forecast[0]) @ Corr_forecast[0] @ np.diag(vol_forecast[0])

        if step > 1:
            for i in range(step):
                Q[i] = A0 + (A + B) * Q[i - 1]
                diag_Q = np.sqrt(np.diag(Q[i]))
                diag_Q = np.diag(diag_Q ** -1)
                Corr_forecast[i] = diag_Q @ Q[i] @ diag_Q
                Cov_forecast[i] = np.diag(vol_forecast[i]) @ Corr_forecast[i] @ np.diag(vol_forecast[i])

        return {"Conditional correlation": Corr_forecast, "Conditional covariance": Cov_forecast, "Mean": mean_forecast}

    def conditional_predict(self, params, partial_views, cov_forecast):
        """
        Given the estimated covariance matrix, we can input a partial view on the realization of one or several assets
        and ask the model to predict what is the conditional mean of the rest of the assets
        :param params:
        :param partial_views: input asset returns
        :param cov_forecast: estimated covariance matrix
        :return: conditional mean of the other asset returns given partial views
        """
        # Partial views needs to be DataFrame with columns = variable names and index = Date
        partial_views = pd.DataFrame(partial_views).T
        final_paths = pd.DataFrame(index=partial_views.index, columns=self.y.columns)
        # Get the index of the variables in partial views
        index = []
        cols = final_paths.columns
        for i in partial_views.columns:
            index.append(cols.get_loc(i))

        full_index = np.arange(self._n_dim)
        # The index of the remaining variables needed to be predicted
        rest_index = list(set(full_index) - set(index))
        temp_res = self.conditional_cov(params)
        mu = temp_res["Mean"].reshape(self._n_dim)
        partial_views = partial_views - mu[index]
        final_paths[partial_views.columns] = partial_views

        # Get the partial covariance matrix
        partial_cov = cov_forecast[np.ix_(index, index)] # dimension p * p
        rest_diag = cov_forecast[np.ix_(rest_index, index)] # dimension (n-p) * p

        # This is a direct application of multinormal or multistudent t dsitribution, E[X1 | X2] where X2 are the
        # partial views, and we are interested in the prediction of X1 given X2
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution    under conditional distributions
        filtered = rest_diag @ np.linalg.inv(partial_cov) @ np.array(partial_views).reshape(partial_views.shape[1], 1)
        filtered = filtered.reshape((1, len(rest_index)))
        temp_append = pd.DataFrame(filtered, index=partial_views.index, columns=cols[rest_index])
        final_paths[cols[rest_index]] = temp_append
        return final_paths + mu

    def partial_to_full(self, partial_views, params, partial_names=None, overwrite_cov=None, **kwargs):
        if not isinstance(partial_views, pd.DataFrame):
            if not partial_names:
                raise Exception("Partial views must have 'partial_names'")

            partial_views = pd.DataFrame(partial_views)
            partial_views.columns = partial_names

        A0 = self.L0 @ self.L0.T
        A = params[0]
        B = params[1]
        window = len(partial_views)
        cov_forecast = np.zeros((window, self._n_dim, self._n_dim))
        corr_forecast = np.zeros((window, self._n_dim, self._n_dim))
        Q = np.zeros((window, self._n_dim, self._n_dim))
        # if overwrite cov, then you can use any historic covariance matrix as if it was the next period covariance
        # matrix and make prediction
        if overwrite_cov:
            cov_forecast[0] = overwrite_cov["Covariance"]
            corr_forecast[0] = overwrite_cov["Correlation"]
            Q[0] = overwrite_cov["Q"]
        else:
            temp = self.forecast(params)
            corr_forecast[0] = temp["Conditional correlation"][0]
            cov_forecast[0] = temp["Conditional covariance"][0]
            Q[0] = temp["Q"][0]

        final_paths = pd.DataFrame(index=[0], columns=self.y.columns)
        mu = self.conditional_cov(params)["Mean"].reshape(self._n_dim)
        final_paths.iloc[0] = np.array(self.conditional_predict(params, partial_views.iloc[0], cov_forecast[0]))

        # Instead of making multi-step forecast, ustilize the partial views to make multi-step forcast for cov matrix
        if window > 1:
            vol_params = []
            std_resid = np.zeros((window, self._n_dim))
            var_forecast = np.zeros((window, self._n_dim))
            vol_forecast = np.zeros((window, self._n_dim))
            var_forecast[0] = np.diag(cov_forecast[0])
            vol_forecast[0] = np.sqrt(var_forecast[0])
            std_resid[0] = final_paths.iloc[0] - mu
            std_resid[0] = std_resid[0] / vol_forecast[0]
            for i in range(self._n_dim):
                vol_params.append(self.volatility_model()[i].params)
                for j in range(1, window):
                    var_forecast[j] = (vol_params[i]['omega'] + vol_params[i]['beta[1'] * var_forecast[j - 1, i]
                                       + vol_params[i]['alpha[1]'] * std_resid[j - 1, i] ** 2
                                       + vol_params[i]['gamma[1]'] * std_resid[j - 1, i] ** 2 if std_resid[j - 1, i] < 0
                                       else 0 if self.vol_asymmetry else 0 )

                    vol_forecast[j] = np.sqrt(var_forecast[j])
                    temp = np.reshape(np.array(std_resid[j - 1]), (self._n_dim, 1))
                    temp = temp @ temp.T
                    Q[j] = A0 + A * (Q[j - 1] - A0) + B * (temp - A0)
                    diag_Q = np.sqrt(np.diag(Q[j]))
                    diag_Q = np.diag(diag_Q ** -1)
                    corr_forecast[j] = diag_Q @ Q[j] @ diag_Q
                    cov_forecast[j] = np.diag(vol_forecast[j]) @ corr_forecast[j] @ np.diag(vol_forecast[j])
                    final_paths.iloc[j] = np.array(self.conditional_predict(params, partial_views.iloc[j],
                                                                            cov_forecast[j]))
                    std_resid[j] = final_paths.iloc[j] - mu
                    std_resid[j] = std_resid[j] / vol_forecast[j]

        return final_paths

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):
        if start_params is None:
            # random guess
            start_params = [0.85, 0.05]
            if self.err_dist == "StudentsT":
                start_params.append(15)

        fit_result = super(DCC, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs)

        # Create a custom result object that wraps the fit result
        class DCCFitResult:
            def __init__(self, fit_result, model):
                self._fit_result = fit_result
                self._model = model
                self.params = fit_result.params

            def __getattr__(self, name):
                return getattr(self._fit_result, name)

            def forecast(self, step=1):
                """Forecast using the estimated parameters"""
                return self._model.forecast(self.params, step=step)

            def conditional_cov(self):
                """Get conditional covariance using the estimated parameters"""
                return self._model.conditional_cov(self.params)

            def conditional_predict(self, partial_view, cov_forecast):
                return self._model.conditional_predict(self.params, partial_view, cov_forecast)

            def partial_to_full(self, partial_views, partial_names=None, overwrite_cov=None):
                return self._model.partial_to_full(partial_views, self.params, partial_names, overwrite_cov)

        return DCCFitResult(fit_result, self)

