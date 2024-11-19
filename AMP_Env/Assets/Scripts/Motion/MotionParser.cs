using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace AMP
{
    public abstract class MotionParser : MonoBehaviour
    {
        public UnityEngine.Object motionFile;
        public abstract List<MotionFrameData> LoadData();
    }
}