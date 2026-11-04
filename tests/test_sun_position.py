"""
Test suite for the sun_position module.

This module tests the solar position calculations to ensure accuracy.
"""

import unittest
import numpy as np
import datetime


class TestSolarPosition(unittest.TestCase):
    """Test solar position calculations."""

    def test_solar_position_noon(self):
        """Test solar position calculation at solar noon."""
        from solweig_gpu.sun_position import sun_position
        
        # Test at solar noon in summer
        # Austin, TX: 30.27°N, 97.74°W on summer day
        time_dict = {
            'year': 2020,
            'month': 7,
            'day': 18,
            'hour': 12,
            'min': 0,
            'sec': 0,
            'UTC': -5
        }
        location_dict = {
            'latitude': 30.27,
            'longitude': -97.74,
            'altitude': 0
        }
        
        sun = sun_position(time_dict, location_dict)
        
        # Should return a dict with zenith and azimuth
        self.assertIsInstance(sun, dict)
        self.assertIn('zenith', sun)
        self.assertIn('azimuth', sun)
        # At noon in summer, zenith should be low (high altitude)
        self.assertLess(sun['zenith'], 30.0, "Solar zenith should be low at summer noon")

    def test_solar_position_sunrise(self):
        """Test solar position calculation at sunrise."""
        from solweig_gpu.sun_position import sun_position
        
        # Austin, TX at sunrise (approximately 6:30 AM)
        time_dict = {
            'year': 2020,
            'month': 7,
            'day': 18,
            'hour': 6,
            'min': 30,
            'sec': 0,
            'UTC': -5
        }
        location_dict = {
            'latitude': 30.27,
            'longitude': -97.74,
            'altitude': 0
        }
        
        sun = sun_position(time_dict, location_dict)
        
        # At sunrise, zenith should be close to 90 (altitude close to 0)
        self.assertIsInstance(sun, dict)
        self.assertTrue(75 <= sun['zenith'] <= 95, f"Zenith {sun['zenith']} should be near 90 at sunrise")

    def test_solar_position_sunset(self):
        """Test solar position calculation at sunset."""
        from solweig_gpu.sun_position import sun_position
        
        # Austin, TX at sunset (approximately 8:00 PM)
        time_dict = {
            'year': 2020,
            'month': 7,
            'day': 18,
            'hour': 20,
            'min': 0,
            'sec': 0,
            'UTC': -5
        }
        location_dict = {
            'latitude': 30.27,
            'longitude': -97.74,
            'altitude': 0
        }
        
        sun = sun_position(time_dict, location_dict)
        
        # At sunset, zenith should be close to 90 (altitude close to 0)
        self.assertIsInstance(sun, dict)
        self.assertTrue(75 <= sun['zenith'] <= 95, f"Zenith {sun['zenith']} should be near 90 at sunset")

    def test_solar_azimuth_range(self):
        """Test that solar azimuth is within valid range [0, 360]."""
        from solweig_gpu.sun_position import sun_position
        
        # Test multiple times of day
        location_dict = {
            'latitude': 30.27,
            'longitude': -97.74,
            'altitude': 0
        }
        
        for hour in [6, 9, 12, 15, 18]:
            time_dict = {
                'year': 2020,
                'month': 7,
                'day': 18,
                'hour': hour,
                'min': 0,
                'sec': 0,
                'UTC': -5
            }
            sun = sun_position(time_dict, location_dict)
            # Azimuth should be in valid range
            self.assertTrue(0 <= sun['azimuth'] <= 360, 
                          f"Azimuth {sun['azimuth']} out of range [0, 360] at hour {hour}")

    def test_solar_altitude_range(self):
        """Test that solar altitude is within valid range [-90, 90]."""
        from solweig_gpu.sun_position import sun_position
        
        # Test multiple times including night
        location_dict = {
            'latitude': 30.27,
            'longitude': -97.74,
            'altitude': 0
        }
        
        for hour in [0, 6, 12, 18, 23]:
            time_dict = {
                'year': 2020,
                'month': 7,
                'day': 18,
                'hour': hour,
                'min': 0,
                'sec': 0,
                'UTC': -5
            }
            sun = sun_position(time_dict, location_dict)
            # Zenith should be in valid range [0, 180], which means altitude in [-90, 90]
            altitude = 90 - sun['zenith']
            self.assertTrue(-90 <= altitude <= 90, 
                          f"Altitude {altitude} out of range [-90, 90] at hour {hour}")


class TestDayLength(unittest.TestCase):
    """Test day length calculations."""

    def test_summer_solstice(self):
        """Test day length calculation at summer solstice."""
        from solweig_gpu.solweig import daylen
        import torch
        
        # Summer solstice (approximately day 172)
        DOY = torch.tensor(172.0)
        # Austin, TX latitude
        XLAT = torch.tensor(30.27)
        
        DAYL, DEC, SNDN, SNUP = daylen(DOY, XLAT)
        
        # Day length should be greater than 12 hours in northern hemisphere summer
        self.assertGreater(DAYL.item(), 12.0)
        # Declination should be positive (northern hemisphere summer)
        self.assertGreater(DEC.item(), 0.0)

    def test_winter_solstice(self):
        """Test day length calculation at winter solstice."""
        from solweig_gpu.solweig import daylen
        import torch
        
        # Winter solstice (approximately day 355)
        DOY = torch.tensor(355.0)
        # Austin, TX latitude
        XLAT = torch.tensor(30.27)
        
        DAYL, DEC, SNDN, SNUP = daylen(DOY, XLAT)
        
        # Day length should be less than 12 hours in northern hemisphere winter
        self.assertLess(DAYL.item(), 12.0)
        # Declination should be negative (northern hemisphere winter)
        self.assertLess(DEC.item(), 0.0)

    def test_equinox(self):
        """Test day length calculation at equinox."""
        from solweig_gpu.solweig import daylen
        import torch
        
        # Spring equinox (approximately day 80)
        DOY = torch.tensor(80.0)
        # Austin, TX latitude
        XLAT = torch.tensor(30.27)
        
        DAYL, DEC, SNDN, SNUP = daylen(DOY, XLAT)
        
        # Day length should be approximately 12 hours at equinox
        self.assertAlmostEqual(DAYL.item(), 12.0, delta=0.5)
        # Declination should be close to zero
        self.assertAlmostEqual(DEC.item(), 0.0, delta=5.0)


class TestSunTimeOffset(unittest.TestCase):
    """The sun is placed half a native step before each metfile row (30 min for hourly data)."""

    # Mumbai on 2024-05-29 (DOY 150), Asia/Kolkata
    LOCATION = {'longitude': 72.85, 'latitude': 19.07, 'altitude': 3.}
    UTC = 5.5

    def _sun(self, times):
        from solweig_gpu.sun_position import Solweig_2015a_metdata_noload
        met = np.array([[2024, 150, h, m] for h, m in times], dtype=float)
        _, altitude, azimuth, *_ = Solweig_2015a_metdata_noload(met, self.LOCATION, self.UTC)
        return altitude[0], azimuth[0]

    def _sun_at(self, hour, minute):
        from solweig_gpu.sun_position import sun_position
        time = {'year': 2024, 'month': 5, 'day': 29, 'hour': hour, 'min': minute, 'sec': 0, 'UTC': self.UTC}
        sun = sun_position(time, self.LOCATION)
        return 90. - float(np.asarray(sun['zenith']).squeeze()), float(np.asarray(sun['azimuth']).squeeze())

    def test_selected_hours_match_a_full_day(self):
        # preprocess(hours=[4, 9, 12]) keeps only the 09:30, 14:30 and 17:30 IST rows. Each
        # must get the sun of the same row in a full day, not one shifted by half the 5 h gap.
        day = [(h, 30) for h in range(24)]
        full_alt, full_azi = self._sun(day)
        for picked in ([(9, 30), (14, 30), (17, 30)], [(14, 30)]):
            alt, azi = self._sun(picked)
            rows = [day.index(t) for t in picked]
            np.testing.assert_allclose(alt, full_alt[rows])
            np.testing.assert_allclose(azi, full_azi[rows])

    def test_hourly_rows_use_the_middle_of_the_hour(self):
        alt, azi = self._sun([(9, 30), (14, 30), (17, 30)])
        for i, (hour, minute) in enumerate([(9, 0), (14, 0), (17, 0)]):
            ref_alt, ref_azi = self._sun_at(hour, minute)
            self.assertAlmostEqual(alt[i], ref_alt, places=6)
            self.assertAlmostEqual(azi[i], ref_azi, places=6)

    def test_half_hourly_rows_keep_a_15_minute_offset(self):
        times = [(h, m) for h in range(8, 16) for m in (0, 30)]
        alt, azi = self._sun(times)
        for i, (hour, minute) in enumerate(times):
            ref_alt, ref_azi = self._sun_at(*divmod(hour * 60 + minute - 15, 60))
            self.assertAlmostEqual(alt[i], ref_alt, places=6)
            self.assertAlmostEqual(azi[i], ref_azi, places=6)


if __name__ == '__main__':
    unittest.main()
