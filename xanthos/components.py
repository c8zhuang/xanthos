"""
Components for use in model configurations.

@author   Chris R. Vernon , lixi729
@email:   chris.vernon@pnnl.gov; xinya.li@pnl.gov
@Project: Xanthos 2.0

License:  BSD 2-Clause, see LICENSE and DISCLAIMER files

Copyright (c) 2017, Battelle Memorial Institute
"""

import numpy as np
import time

import xanthos.data_reader.data_load as fetch
import xanthos.utils.general as helper
import xanthos.utils.math as um
from xanthos.data_writer.out_writer import OUTWriter
from xanthos.diagnostics.aggregation import Aggregation
from xanthos.diagnostics.diagnostics import Diagnostics
from xanthos.diagnostics.time_series import TimeSeriesPlot
from xanthos.accessible.accessible import AccessibleWater
from xanthos.hydropower.potential import HydropowerPotential
from xanthos.hydropower.actual import HydropowerActual


class Components:
    """
    Components for use in model configurations.

    @author   Chris R. Vernon , lixi729
    @email:   chris.vernon@pnnl.gov; xinya.li@pnl.gov
    @Project: Xanthos 2.0

    License:  BSD 2-Clause, see LICENSE and DISCLAIMER files

    Copyright (c) 2017, Battelle Memorial Institute
    """

    def __init__(self, config):

        self.s = config

        # import desired modules for PET, Runoff, and Routing
        self.import_core()

        # get data required for each module
        if self.s.pet_module == 'hargreaves':

            # average monthly temperature in degree C
            self.temp = None #fetch.load_climate_data(self.s.TemperatureFile, self.s.ncell, self.s.nmonths,
                                                #varname=self.s.TempVarName)

            # monthly average of daily temperature range in degree c
            self.dtr = None #fetch.load_climate_data(self.s.DailyTemperatureRangeFile, self.s.ncell, self.s.nmonths,
                                               #varname=self.s.DTRVarName, neg_to_zero=True)

        elif self.s.pet_module == 'penman-monteith':

            # monthly average of minimum daily temperature in degree C
            self.tmin = fetch.load_climate_data(self.s.DailyTemperatureRangeFile, self.s.ncell, self.s.nmonths,
                                                varname=self.s.TminVarName)

            # relative humidity in %
            self.rhs = fetch.load_climate_data(self.s.relative_humidity_file, self.s.ncell, self.s.nmonths,
                                                varname=self.s.RhsVarName)

            # monthly average surface air temperature degree C
            self.tair = fetch.load_climate_data(self.s.tair_file, self.s.ncell, self.s.nmonths,
                                               varname=self.s.TairVarName)

            # relative humidity in %
            self.rlds = fetch.load_climate_data(self.s.rlds_file, self.s.ncell, self.s.nmonths,
                                               varname=self.s.rlds_var)
            # rsds
            self.rsds = fetch.load_climate_data(self.s.rsds_file, self.s.ncell, self.s.nmonths,
                                                varname=self.s.rsds_var)

            # literature derived parameters for ET
            self.pm_params = np.genfromtxt(self.s.pm_param_file, delimiter=',')

            # albedo
            self.alpha = np.genfromtxt(self.s.albedo_file, delimiter=',')

            # leaf area index average
            self.lai = np.genfromtxt(self.s.lai_file, delimiter=',')

            # leaf area index minimum
            self.laimin = np.genfromtxt(self.s.laimin_file, delimiter=',')

            # leaf area index maximum
            self.laimax = np.genfromtxt(self.s.laimax_file, delimiter=',')

            # land cover data
            self.lct = np.load(self.s.land_cover_file)

            # elevation data
            self.elev = np.load(self.s.elev_file)


        if self.s.runoff_module == 'gwam':

            # monthly precipitation mm/mth
            self.precip = fetch.load_climate_data(self.s.PrecipitationFile, self.s.ncell, self.s.nmonths, varname=self.s.PrecipVarName)

            # monthly average daily temperature
            self.temp = fetch.load_climate_data(self.s.TemperatureFile, self.s.ncell, self.s.nmonths, varname=self.s.TempVarName)



        # climate data
        if self.s.climate and self.s.runoff_module == 'hargreaves':

            self.precip = fetch.load_climate_data(self.s.PrecipitationFile, self.s.ncell, self.s.nmonths, varname=self.s.PrecipVarName)
            self.temp = fetch.load_climate_data(self.s.TemperatureFile, self.s.ncell, self.s.nmonths, varname=self.s.TempVarName)



        # reference data
        self.ref = fetch.LoadReferenceData(self.s)
        self.coords = self.ref.coords
        self.latitude = np.copy(self.coords[:, 2])
        self.lat_radians = np.radians(self.latitude) # * np.pi / 180.
        self.yr_imth_dys = helper.set_month_arrays(self.s.nmonths, self.s.StartYear, self.s.EndYear)
        self.map_index = um.sub2ind([self.s.ngridrow, self.s.ngridcol], self.coords[:, 4].astype(int) - 1, self.coords[:, 3].astype(int) - 1)

        sft = helper.calc_sinusoidal_factor(self.yr_imth_dys)
        self.solar_dec = sft[0]
        self.dr = sft[1]

        # pet
        if self.s.pet_module == 'hargreaves':
            self.mth_temp_pet = None
            self.mth_dtr_pet = None
            self.pet_t = None
            self.pet_out = None #np.zeros_like(self.precip)

        elif self.s.pet_module == 'penman-monteith':
            self.pet_out = np.zeros_like(self.precip)
            pass

        # runoff
        if self.s.runoff_module == 'gwam':
            self.soil_moisture = None
            self.grid_area = None
            self.sm_prev = None

        elif self.s.runoff_module == 'abcd':
            pass

        # routing
        if self.s.routing_module == 'mrtm':
            self.flow_dist = None
            self.flow_dir = None
            self.instream_flow = None
            self.str_velocity = None
            self.dsid = None
            self.upid = None
            self.um = None
            self.routing_timestep_hours = 3 * 3600
            self.chs_prev = None

        # outputs
        self.PET = np.zeros(shape=(self.s.ncell, self.s.nmonths))
        self.AET = np.zeros(shape=(self.s.ncell, self.s.nmonths))
        self.Q = np.zeros(shape=(self.s.ncell, self.s.nmonths))
        self.Sav = np.zeros(shape=(self.s.ncell, self.s.nmonths))
        self.ChStorage = np.zeros(shape=(self.s.ncell, self.s.nmonths))
        self.Avg_ChFlow = np.zeros(shape=(self.s.ncell, self.s.nmonths))

        # simulation
        self.P = None
        self.T = None
        self.D = None
        self.mth_solar_dec = None
        self.mth_dr = None
        self.mth_days = None

        # outputs
        self.q = None
        self.ac = None

        # prep modules
        self.prep_runoff()
        self.prep_routing()

    def import_core(self):
        """
        Import desired core modules.
        """
        global pet_mod
        global runoff_mod
        global routing_mod

        # import desired module for PET
        if self.s.pet_module == 'hargreaves':
            import xanthos.pet.hargreaves as pet_mod

        elif self.s.pet_module == 'penman-monteith':
            import xanthos.pet.penman_monteith as pet_mod

        # import desired module for Runoff
        if self.s.runoff_module == 'gwam':
            import xanthos.runoff.gwam as runoff_mod

        elif self.s.runoff_module == 'abcd':
            import xanthos.runoff.abcd as runoff_mod

        # import desired module for Routing
        if self.s.routing_module == 'mrtm':
            import xanthos.routing.mrtm as routing_mod

    def prep_arrays(self, nm=None):
        """
        Prepare arrays.
        """
        if nm is None:
            self.P = np.copy(self.precip)  # keep nan in P
            self.T = np.nan_to_num(self.temp)  # nan to zero
            self.D = np.nan_to_num(self.dtr)  # nan to zero

        else:
            self.P = np.copy(self.precip[:, nm])  # keep nan in P
            self.T = np.nan_to_num(self.temp[:, nm])  # nan to zero
            self.D = np.nan_to_num(self.dtr[:, nm])  # nan to zero

    def prep_pet(self, nm):
        """
        Prepare PET for processing.

        @:param nm:                     time-step number

        @:return mth_solar_dec:         solar declination for the target step
        @:return mth_dr:                inverse relative distance Earth-Sun for the target step
        @:return mth_days:              number of days in the target month
        @:return temp_pet:              PET for the target month
        @:return dtr_pet:               Daily temperature range for the target month
        """
        if self.s.pet_module == 'hargreaves':

            self.mth_solar_dec = np.copy(self.solar_dec[nm])
            self.mth_dr = np.copy(self.dr[nm])
            self.mth_days = np.copy(self.yr_imth_dys[nm, 2])
            self.mth_temp_pet = np.nan_to_num(self.temp[:, nm])
            self.mth_dtr_pet = np.nan_to_num(self.dtr[:, nm])

        elif self.s.pet_module == 'penman-monteith':
            pass

    def calculate_pet(self, nm=None):
        """
        Calculate monthly potential evapo-transpiration.
        """
        if self.s.pet_module == 'hargreaves':

            self.pet_t = pet_mod.calculate_pet(self.mth_temp_pet, self.mth_dtr_pet, self.lat_radians, self.mth_solar_dec, self.mth_dr, self.mth_days)

        elif self.s.pet_module == 'penman-monteith':
            pass

    def prep_runoff(self):
        """
        Assign max soil moisture (mm/month) [2] to Sm.  For historic data use 0.5 * sm to an initial value to pass to
        runoff model. If future mode, read values from historical file.
        """
        if self.s.runoff_module == 'gwam':
            # create a matrix (MSMC: Maximum Soil Moisture Capacity) with all data
            # 1: area; 2: region; 3: Max Soil Moisture (mm/month)
            msmc = fetch.load_soil_moisture(self.ref, self.s.ncell)

            # harmonized grid area
            self.grid_area = np.copy(msmc[:, 0])

            # maximum soil moisture
            self.soil_moisture = np.copy(msmc[:, 2])

            # load soil moisture file if running future
            if self.s.HistFlag == "True":
                self.sm_prev = 0.5 * self.soil_moisture

            else:
                self.sm_prev = fetch.load_soil_data(self.s)

        elif self.s.runoff_module == 'abcd':
            # self.P = np.copy(self.precip)  # keep nan in P
            # self.T = np.nan_to_num(self.temp)  # nan to zero
            # self.D = np.nan_to_num(self.dtr)  # nan to zero
            # self.pet_out = np.zeros_like(self.precip)
            pass

    def calculate_runoff(self, nm=None):
        """
        Calculate runoff.

        Runoff takes a feedback of soil moisture content (sm_prev).  This is updated for each iteration of a month.
        """
        if self.s.runoff_module == 'gwam':

            rg = runoff_mod.runoffgen(self.pet_t, self.P, self.s, self.soil_moisture, self.sm_prev)

            self.PET[:, nm], self.AET[:, nm], self.Q[:, nm], self.Sav[:, nm] = rg

        elif self.s.runoff_module == 'abcd':

            # np.save('/users/ladmin/Desktop/temp_xanthos/basin_ids.npy', self.ref.basin_ids)

            # f_pet = '/users/ladmin/Google Drive/work/xanthos/new_pet/PET_output/penman_monteith_pet_watch_static1970lc_1951_2005.npy'
            # f_pet = '/users/ladmin/Desktop/pet_pm_result.npy'
            # f_pet = '/users/ladmin/Google Drive/work/xanthos/new_pet/global_pet_1971_1990_staticland1971.npy'
            f_pet = '/users/ladmin/Desktop/penman_monteith/penman_monteith_watch_1971_1990_1970lc.npy'

            # f_pr = '/users/ladmin/Google Drive/work/xanthos/new_pet/Python_input/watch_pr_1951_2005.npy'
            # f_tmin = '/users/ladmin/Google Drive/work/xanthos/new_pet/Python_input/watch_tmin_1951_2005.npy'

            f_pr = '/users/ladmin/Desktop/watch_pm_abcd_inputs_1971_1990/prec_1971_1990_mmmth.npy'
            f_tmin = '/users/ladmin/Desktop/watch_pm_abcd_inputs_1971_1990/tmin_1971_1990.mat'
            self.s.nmonths = 240

            import scipy.io as sio
            self.pet_out = np.load(f_pet)
            # self.precip = sio.loadmat(f_pr)['prec_1971_1990']
            self.precip = np.load(f_pr)
            tmin = sio.loadmat(f_tmin)['TMIN_1971_1990']

            # run all basins at once in parallel
            rg = runoff_mod.abcd_execute(n_basins=235, basin_ids=self.ref.basin_ids,
                                         pet=self.pet_out, precip=self.precip, tmin=np.nan_to_num(tmin),
                                         calib_file=self.s.calib_file, n_months=self.s.nmonths,
                                         spinup_factor=self.s.SpinUp, jobs=self.s.ro_jobs)

            self.PET, self.AET, self.Q, self.Sav = rg

    def prep_routing(self):
        """
        Prepare routing arrays.
        """
        if self.s.routing_module == 'mrtm':

            self.flow_dist = fetch.load_routing_data(self.s.FlowDis, self.s.ngridrow, self.s.ngridcol,
                                                     self.map_index, rep_val=1000)
            self.flow_dir = fetch.load_routing_data(self.s.FlowDir, self.s.ngridrow, self.s.ngridcol,
                                                    self.map_index)
            self.instream_flow = np.zeros((self.s.ncell,), dtype=float)
            self.str_velocity = fetch.load_routing_data(self.s.strm_veloc, self.s.ngridrow, self.s.ngridcol,
                                                        self.map_index, rep_val=0)
            self.dsid = routing_mod.downstream(self.coords, self.flow_dir, self.s)
            self.upid = routing_mod.upstream(self.coords, self.dsid, self.s)
            self.um = routing_mod.upstream_genmatrix(self.upid)
            self.chs_prev = fetch.load_chs_data(self.s)

            # if user is providing a custom runoff file
            if self.s.alt_runoff is not None:
                self.Q = np.load(self.s.alt_runoff)

    def calculate_routing(self, nm):
        """
        Calculate routing.  Routing takes a simulated runoff (Q) from the runoff output and
        previous channel storage (chs_prev) from previous channel storage.
        """
        if self.s.routing_module == 'mrtm':
            sr = routing_mod.streamrouting(self.flow_dist, self.chs_prev, self.instream_flow, self.str_velocity,
                                           self.Q[:, nm], self.ref.area, self.yr_imth_dys[nm, 2],
                                           self.routing_timestep_hours, self.um)

            self.ChStorage[:, nm], self.Avg_ChFlow[:, nm], self.instream_flow = sr

    def simulation(self, pet=True, pet_num_steps=0, pet_step='month',
                   runoff=True, runoff_num_steps=0, runoff_step='month',
                   routing=True, routing_num_steps=0, routing_step='month',
                   notify='simulation'):
        """
        Run model simulation for a defined configuration.

        :param num_steps:           The number of time steps to process (INT)
        :param pet:                 True if running PET, False if embedded in runoff model
        :param runoff:              True if running Runoff, False if not
        :param runoff_step:         The time unit as a string; if None the runoff model iterates internally, else 'month'
        :param routing_num_steps:   The number of steps for to run the routing module; different on spin-up, same as runoff
                                    when running normal.  Specified in months in the config file.
        :param routing:             True if running Routing, False if not
        :param routing_step:        The time unit as a string; if None the routing model iterates internally, else 'month'
        :param notify:              A string that is used to add to log print that describes whether the simulation is
                                    spin-up or regular
        """
        # pass simulation if there are no steps to process
        if (pet_num_steps + runoff_num_steps + routing_num_steps) == 0:
            pass

        else:

            if (pet_step == 'month') and (runoff_step == 'month') and (routing_step == 'month'):

                print("---{} in progress...".format(notify))
                t0 = time.time()

                if pet:

                    print("\tProcessing PET...")
                    t = time.time()

                    for nm in range(pet_num_steps):
                        # set up climate data for processing
                        self.prep_arrays(nm)

                        # set up PET data for processing
                        self.prep_pet(nm)

                        # calculate pet
                        self.calculate_pet()

                    print("\tPET processed in {} seconds---".format(time.time() - t))

                if runoff:

                    print("\tProcessing Runoff...")
                    t = time.time()

                    for nm in range(runoff_num_steps):

                        # calculate runoff and generate monthly potential ET, actual ET, runoff, and soil moisture
                        if runoff:
                            self.calculate_runoff(nm)

                            # update soil moisture (sav) array for next step
                            self.sm_prev = np.copy(self.Sav[:, nm])

                    print("\tRunoff processed in {} seconds---".format(time.time() - t))

                # channel storage, avg. channel flow (m^3/sec), instantaneous channel flow (m^3/sec)
                if routing:

                    print("\tProcessing Routing...")
                    t = time.time()

                    for nm in range(routing_num_steps):
                        self.calculate_routing(nm)

                        # update channel storage (chs) array for next step
                        self.chs_prev = np.copy(self.ChStorage[:, nm])

                    print("\tRouting processed in {} seconds---".format(time.time() - t))

                print("---{0} has finished successfully: {1} seconds ---".format(notify, time.time() - t0))

            elif (pet_step == 'month') and (runoff_step is None) and (routing_step == 'month'):

                print("---{} in progress... ".format(notify))
                t0 = time.time()

                # set up climate data for processing
                # self.prep_arrays()

                # calculate PET
                if pet:
                    print("\tProcessing PET...")
                    t = time.time()

                    for nm in range(pet_num_steps):
                        # set up PET data for processing
                        self.prep_pet(nm)

                        # calculate pet
                        self.calculate_pet()

                        # archive pet month in array
                        self.pet_out[:, nm] = self.pet_t

                    print("\tPET processed in {} seconds---".format(time.time() - t))

                # calculate runoff for all basins all months
                if runoff:
                    print("\tProcessing Runoff...")
                    t = time.time()

                    self.calculate_runoff()

                    print("\tRunoff processed in {} seconds---".format(time.time() - t))

                # process routing
                if routing:

                    print("\tProcessing Routing...")
                    t = time.time()

                    for nm in range(routing_num_steps):
                        # channel storage, avg. channel flow (m^3/sec), instantaneous channel flow (m^3/sec)
                        self.calculate_routing(nm)

                        # update channel storage (chs) arrays for next step
                        self.chs_prev = np.copy(self.ChStorage[:, nm])

                    print("\tRouting processed in {} seconds---".format(time.time() - t))

                print("---{0} has finished successfully: {1} seconds ---".format(notify, time.time() - t0))

    def accessible_water(self):
        """
        Run accessible water module
        """
        if self.s.CalculateAccessibleWater:
            print("---Start Accessible Water:")
            t0 = time.time()

            AccessibleWater(self.s, self.ref, self.Q)

            print("---Accessible Water has finished successfully: %s seconds ------" % (time.time() - t0))

    def hydropower_potential(self):
        """
        Run hydropower potential module.
        """
        if self.s.CalculateHydropowerPotential:
            print("---Start Hydropower Potential:")
            t0 = time.time()

            HydropowerPotential(self.s, self.Avg_ChFlow)

            print("---Hydropower Potential has finished successfully: %s seconds ------" % (time.time() - t0))

    def hydropower_actual(self):
        """
        Run hydropower actual module.
        """
        if self.s.CalculateHydropowerActual:
            print("---Start Hydropower Actual:")
            t0 = time.time()

            HydropowerActual(self.s, self.Avg_ChFlow)

            print("---Hydropower Actual has finished successfully: %s seconds ------" % (time.time() - t0))

    def diagnostics(self):
        """
        Run diagnostics.
        """
        if self.s.PerformDiagnostics:
            print("---Start Diagnostics:")
            t0 = time.time()

            Diagnostics(self.s, self.Q, self.ref)

            print("---Diagnostics has finished successfully: %s seconds ------" % (time.time() - t0))

    def output_simulation(self):
        """
        Output simulation results.
        """
        print "---Output simulation results:"
        t0 = time.time()

        self.q, self.ac = OUTWriter(self.s, self.ref.area, self.PET, self.AET, self.Q, self.Sav, self.ChStorage,
                                    self.Avg_ChFlow)

        print("---Output finished: %s seconds ---" % (time.time() - t0))

    def aggregate_outputs(self):
        """
        Aggregation by Basin, Country, and/or Region.
        """
        if self.s.AggregateRunoffBasin > 0 or self.s.AggregateRunoffCountry > 0 or self.s.AggregateRunoffGCAMRegion > 0:
            print("---Start Aggregation:")
            t0 = time.time()

            Aggregation(self.s, self.ref, self.q)

            print("---Aggregation has finished successfully: %s seconds ------" % (time.time() - t0))

    def plots(self):
        """
        Create time series plots.
        """
        if self.s.CreateTimeSeriesPlot:
            print("---Creating Time Series Plots:")
            t0 = time.time()

            TimeSeriesPlot(self.s, self.q, self.ac, self.ref)

            print("---Plots has finished successfully: %s seconds ------" % (time.time() - t0))

    def process(self):
        """
        Inputs:
        precip:      monthly precipitation averages [ncell x nmonth]
        temp:        monthly temperature averages [ncell x nmonth]
        dtr:         monthly average daily temperature range [ncell x nmonth]
        initstorage: initial channel storage
        unitflag:    1= output in km^3/yr.  0= output in mm/yr
        histflag:    1= running historical hydro; 0= running future hydro
        startmonth:  month of year for the first month in the dataset.  1=Jan, 2=Feb, etc.

        Outputs:
        q:           monthly runoff [ncell x nmonth]
        pet:         monthly potential evapo-transpiration
        aet:         monthly actual evapo-transpiration
        s:           monthly channel storage
        Avg_ChFlow:  monthly average stream flow [ncell x nmonth]
        ChStorage:   month-end channel storage [ncell x nmonth]
        """
        pass
