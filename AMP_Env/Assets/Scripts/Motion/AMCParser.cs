using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace AMP
{
    public class AMCParser : MotionParser
    {
        public override List<MotionFrameData> LoadData()
        {
            List<MotionFrameData> frameData = new List<MotionFrameData>();

            string text = Utils.ReadTextFile(motionFile);
            if (string.IsNullOrEmpty(text))
            {
                Debug.LogWarning($"Wrong text {motionFile}");
                return null;
            }

            string[] lines = text.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None)
                                .Where(s => !string.IsNullOrWhiteSpace(s))
                                .ToArray();
            frameData = new List<MotionFrameData>();

            MotionFrameData currentFrameData = null;
            foreach (string line in lines.Skip(3))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                if (int.TryParse(line, out int frameNumber))
                {
                    currentFrameData = new MotionFrameData();
                    frameData.Add(currentFrameData);
                }
                else
                {
                    var parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 2)
                    {
                        string jointName = parts[0];
                        List<float> values = parts.Skip(1).Select(float.Parse).ToList();
                        currentFrameData.JointData[jointName] = values;
                    }
                }
            }

            return frameData;
        }
    }
}