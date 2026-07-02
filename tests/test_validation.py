import unittest

from emissions_ml.validation import expanding_window_folds, pre_holdout_years


class RollingOriginValidationTest(unittest.TestCase):
    def setUp(self):
        self.folds = expanding_window_folds()

    def test_training_years_always_precede_validation_year(self):
        for fold in self.folds:
            self.assertTrue(all(year < fold.validation_year for year in fold.training_years))

    def test_training_window_expands_with_each_fold(self):
        training_windows = [fold.training_years for fold in self.folds]

        self.assertEqual(training_windows[0], (2011,))
        self.assertEqual(training_windows[1], (2011, 2012))
        self.assertEqual(training_windows[2], (2011, 2012, 2013))

    def test_exactly_one_year_is_used_for_validation_in_each_fold(self):
        validation_years = [fold.validation_year for fold in self.folds]

        self.assertEqual(validation_years, [2012, 2013, 2014])
        self.assertEqual(len(validation_years), len(set(validation_years)))

    def test_unsorted_validation_years_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            expanding_window_folds(validation_years=(2013, 2012, 2014))

    def test_duplicate_validation_years_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            expanding_window_folds(validation_years=(2012, 2013, 2013))

    def test_holdout_year_never_appears_in_validation_folds(self):
        for fold in self.folds:
            self.assertNotIn(2015, fold.training_years)
            self.assertNotEqual(fold.validation_year, 2015)

    def test_final_development_set_includes_years_before_holdout(self):
        self.assertEqual(pre_holdout_years(), (2011, 2012, 2013, 2014))


if __name__ == "__main__":
    unittest.main()
