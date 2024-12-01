using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json;
using System.Linq;
using System.IO;

namespace AMP
{
    public class DeepMimicMotionParser : MotionParser
    {
        #region Class for parse json

        [Serializable]
        public class MotionData
        {
            public string Loop;
            public float[,] Frames;
        }
        #endregion

        // {ID, dofs}
        private int[,] dofs = new int[,]
        {
            {-1, 1}, // duration of frame in seconds
            {0, 7}, // root position(3D) +  root rotation(4D)
            {1, 4}, // chest rotation
            {2, 4}, // neck rotation
            {3, 4}, // right hip rotation
            {4, 1}, // right knee rotation
            {5, 4}, // right ankel rotation
            {6, 4}, // right shoulder rotation
            {7, 1}, // right elbow rotation
            {9, 4}, // left hip rotation
            {10, 1}, // left knee rotation
            {11, 4}, // left ankel rotation
            {12, 4}, // left shoulder rotation
            {13, 1}  // left elbow rotation
        };


        public override List<MotionFrameData> LoadData(string motionFilePath)
        {
            List<MotionFrameData> motionFrameData = new List<MotionFrameData>();

            string path = Path.Combine(Utils.GetCurrentPath(), motionFilePath);
            string text = Utils.ReadTextFile(path);

            if (string.IsNullOrEmpty(text))
            {
                Debug.LogWarning($"Wrong path {motionFilePath}");
                return null;
            }

            MotionData motionData = JsonConvert.DeserializeObject<MotionData>(text);
            for (int i = 0; i < motionData.Frames.GetLength(0); i++)
            {
                if (motionData.Frames.GetLength(1) != 44)
                {
                    Debug.LogWarning($"Wrong data, frame data size is {motionData.Frames.GetLength(1)}");
                    break;
                }

                int frame = 1;
                int dofIdx = 1;
                MotionFrameData frameData = new MotionFrameData();
                while (dofIdx < dofs.GetLength(0))
                {
                    List<float> values = Enumerable.Range(frame, dofs[dofIdx, 1])
                                                   .Select(col => motionData.Frames[i, col])
                                                   .ToList();

                    frameData.JointData[dofs[dofIdx, 0]] = values;
                    frame += dofs[dofIdx, 1];
                    dofIdx++;
                }

                motionFrameData.Add(frameData);
            }


            return motionFrameData;
        }
    }
}