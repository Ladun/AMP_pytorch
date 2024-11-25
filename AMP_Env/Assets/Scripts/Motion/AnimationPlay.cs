using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace AMP
{
    public class MotionFrameData
    {
        public Dictionary<string, List<float>> JointData = new Dictionary<string, List<float>>();
    }

    public class AnimationPlay : MonoBehaviour
    {

        public Skeleton skeleton;
        public MotionDatabase motionDatabase;

        [Header("Animation player property")]
        public List<MotionFrameData> frameData;
        public int currentFrame = 0;


        public int totalFrame
        {
            get { return frameData == null? 0 : frameData.Count; }
        }

        public MotionFrameData currentFrameData
        {
            get
            {
                if (frameData == null || frameData.Count <= currentFrame || currentFrame < 0) 
                    return null;
                return frameData[currentFrame];
            }
        }

        public void LoadData(string key)
        {
            if (string.IsNullOrEmpty(key))
                return;
            if (skeleton != null)
                skeleton.CreateSkeleton();

            currentFrame = 0;
            frameData = motionDatabase.GetMotionData(key);

            PlayAnimation();
        }

        public void PlayAnimation()
        {
            if (frameData == null || currentFrame >= frameData.Count || currentFrame < 0)
                return;

            MotionFrameData data = frameData[currentFrame];
            skeleton.SetAnimationData(data);
        }

    }
}