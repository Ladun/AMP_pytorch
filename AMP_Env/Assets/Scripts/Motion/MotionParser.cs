using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace AMP
{
    public abstract class MotionParser
    {
        public abstract List<MotionFrameData> LoadData(string motionFilePath);
    }
}