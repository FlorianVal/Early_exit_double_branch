import unittest
import torch
from src.BranchyResnet import BranchyResNet18

class TestBranchyResNet(unittest.TestCase):
    def setUp(self):
        self.num_aux_heads = 4
        self.model = BranchyResNet18(num_classes=10, num_aux_heads=self.num_aux_heads)

    def test_instance(self):
        self.assertIsInstance(self.model, BranchyResNet18)

    def test_forward_pass(self):
        x = torch.randn(2, 3, 224, 224)
        output = self.model(x)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[0], self.num_aux_heads + 1)  # num_aux_heads auxiliary heads + 1 main head
        self.assertEqual(output.shape[1], 2)  # batch size
        self.assertEqual(output.shape[2], 10)  # num_classes

    def test_set_aux_head_output(self):
        self.model.set_aux_head_output(3)
        self.assertEqual(self.model.aux_head_output, 3)
        output = self.model(torch.randn(2, 3, 224, 224))
        self.assertEqual(output.shape[0], 1) # only one output head
        self.assertEqual(output.shape[1], 2) # batch size
        self.assertEqual(output.shape[2], 10) # num_classes

if __name__ == '__main__':
    unittest.main()
