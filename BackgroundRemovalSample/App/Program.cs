using System;
using System.Diagnostics;
using System.IO;
using Fclp;
using JetBrains.Annotations;
using OpenCvSharp;

namespace BackgroundRemovalSample.App
{
	internal static class Program
	{
		private static void Main(String[] args)
		{
			var parser = new FluentCommandLineParser<Application.ApplicationArguments>();
			parser.Setup(arg => arg.InputImagePath)
				.As('i', "img")
				.SetDefault("image.jpg");

			parser.Setup(arg => arg.OutputImagePath)
				.As('o', "out")
				.SetDefault("out.png");

			parser.Setup(arg => arg.FloodFillTolerance)
				.As('f', "flood-fill-tolerance")
				.SetDefault(0.01);

			parser.Setup(arg => arg.MaskBlurFactor)
				.As('b', "blur")
				.SetDefault(5);

			parser.SetupHelp();

			if (parser.Parse(args).HasErrors)
				parser.HelpOption.ShowHelp(parser.Options);

			new Application(parser.Object).Run();
		}
	}

	internal class Application
	{
		public Application(ApplicationArguments args)
		{
			if (!File.Exists(args.InputImagePath))
				throw new ArgumentException("Input image does not exist.");

			_args = args;
		}

		private readonly ApplicationArguments _args;

		public void Run()
		{
			using (var img = new Mat(_args.InputImagePath))
			{
				var filter = new RemoveBackgroundOpenCvFilter
				{
					FloodFillTolerance = _args.FloodFillTolerance,
					MaskBlurFactor     = _args.MaskBlurFactor
				};

				var sw = new Stopwatch();
				sw.Start();
				using (var result = filter.Apply(img))
				{
					sw.Stop();
					Console.WriteLine($"Run {sw.ElapsedMilliseconds}ms");

					result.SaveImage(_args.OutputImagePath);
				}
			}
		}

		[UsedImplicitly]
		public class ApplicationArguments
		{
			public String InputImagePath     { get; set; }
			public String OutputImagePath    { get; set; }
			public Double FloodFillTolerance { get; set; }
			public Int32  MaskBlurFactor     { get; set; }
		}
	}
}