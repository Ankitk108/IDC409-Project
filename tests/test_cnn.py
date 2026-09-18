"""Backend-independent checks for the experimental CNN representation."""

import unittest

from src.cnn import CLEO_BC, CLEO_ROE, KSFW_BC, KSFW_FS1, cnn_feature_groups


class CNNRepresentationTests(unittest.TestCase):
    def test_sequences_and_scalars_partition_features_once(self):
        features = ["R2", "thrust", *CLEO_BC, *CLEO_ROE, *KSFW_BC, *KSFW_FS1]
        groups = cnn_feature_groups(features)
        flattened = [
            *groups.scalar, *groups.cleo_bc, *groups.cleo_roe,
            *groups.ksfw_bc, *groups.ksfw_fs1,
        ]
        self.assertEqual(set(flattened), set(features))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(groups.scalar, ("R2", "thrust"))
        self.assertEqual(len(groups.cleo_bc), 9)
        self.assertEqual(len(groups.ksfw_bc), 14)

    def test_missing_sequence_feature_is_rejected(self):
        features = ["R2", *CLEO_BC, *CLEO_ROE, *KSFW_BC, *KSFW_FS1]
        features.remove(CLEO_BC[0])
        with self.assertRaises(ValueError):
            cnn_feature_groups(features)


if __name__ == "__main__":
    unittest.main()
