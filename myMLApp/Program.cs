using System;
using MyMLAppML.Model.DataModels;
using Microsoft.ML;

namespace myMLApp
{
    class Program
    {
        static void Main(string[] args)
        {
            ConsumeModel();
        }

        public static void ConsumeModel()
        {
            // Load the model
            MLContext mlContext = new MLContext();

            ITransformer mlModel = mlContext.Model.Load("MLModel.zip", out var modelInputSchema);

            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            // Use the code below to add input data
            var input = new ModelInput();
            //below should be toxic
            //input.SentimentText = "It was really cool";
            // this should be non toxic, below
            input.SentimentText = "Much happier";

            // Try model on sample data
            // True is toxic, false is non-toxic
            ModelOutput result = predEngine.Predict(input);

            Console.WriteLine($"Text: {input.SentimentText} | Prediction: {(Convert.ToBoolean(result.Prediction) ? "Toxic" : "Non Toxic")} sentiment");
        }
    }
}