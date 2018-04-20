using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace BackgroundRemovalSample.App
{
	/// <summary>
	///     Base class for custom OpenCV filters. More convenient than plain static methods.
	/// </summary>
	public abstract class OpenCvFilter
	{
		static OpenCvFilter()
		{
			Cv2.SetUseOptimized(true);
		}

		/// <summary>
		///     Supported depth types of input array.
		/// </summary>
		public abstract IEnumerable<MatType> SupportedMatTypes { get; }

		/// <summary>
		///     Applies filter to <see cref="src" /> and returns result.
		/// </summary>
		/// <param name="src">Source array.</param>
		/// <returns>Result of processing filter.</returns>
		public Mat Apply(Mat src)
		{
			var dst = new Mat();
			ApplyInPlace(src, dst);

			return dst;
		}

		/// <summary>
		///     Applies filter to <see cref="src" /> and writes to <see cref="dst" />.
		/// </summary>
		/// <param name="src">Source array.</param>
		/// <param name="dst">Output array.</param>
		/// <exception cref="ArgumentException">Provided image does not meet the requirements.</exception>
		public void ApplyInPlace(Mat src, Mat dst)
		{
			if (!SupportedMatTypes.Contains(src.Type()))
				throw new ArgumentException("Depth type of provided Mat is not supported");

			ProcessFilter(src, dst);
		}

		/// <summary>
		///     Actual filter.
		/// </summary>
		/// <param name="src">Source array.</param>
		/// <param name="dst">Output array.</param>
		protected abstract void ProcessFilter(Mat src, Mat dst);
	}
}