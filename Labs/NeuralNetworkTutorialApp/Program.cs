using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;

namespace NeuralNetworkTutorialApp
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] layerSizes = new int[3] { 3, 4, 1 };
            TransferFunction[] TFuncs = new TransferFunction[3] {TransferFunction.None,
                                                               TransferFunction.Sigmoid,
                                                               TransferFunction.Linear};

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, TFuncs);

            double[] input = new double[] { 4, 6, 8 };//, new double[] {4, 7, 5}, new double[] {7, 4, 8}, new double[] {6, 7, 5}, new double[] {7, 7, 8}};

            /*for(int i = 0; i < input.GetUpperBound(0); i++)
            {
                input[i] = new double[3];
                for (int j = 0; j < input[i].Length; j++)
                {

                }
            }*/

            double[] desired = new double[] { -0.86 };//, new double[] {0.15}, new double[] {0.72 }, new double[] {0.53 }, new double[] { 0.44 } };
            double[] output = new double[1];
            

            double error = 0.0;

            for(int i = 0; i < 10; i++)
            {
               
                    error = bpn.Train(ref input, ref desired, 0.15, 0.1);
                    bpn.Run(ref input, out output);
                    if (i % 1 == 0)
                        Console.WriteLine("Iteration {0}: \n\t Input {1:0.000} {2:0.000} {3:0.000} Output {4:0.000} error{5:0.000}", i, input[0], input[1], input[2], output[0], error);
                    /*for (int k = 0; k < 4; k++)
                        Console.WriteLine("{0:0.000}", bpn.layerOtput[0][k]);*/
                
            }

            Console.ReadKey();
        }
    }
}
