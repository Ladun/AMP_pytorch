using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static AMP.DeepMinicSkeleton;

namespace AMP
{
    public abstract class Skeleton : MonoBehaviour
    {
        public class Observastion
        {
            public List<Vector3> positions = new List<Vector3>();
            public List<Vector3> normals = new List<Vector3>();
            public List<Vector3> tangents = new List<Vector3>();
            public List<Vector3> linearVels = new List<Vector3>();
            public List<Vector3> angularVels = new List<Vector3>();

            public int GetObsSize()
            {
                // positions.Count == normals.Count == tangents.Count == ...
                return positions.Count * 5;
            }

            public void Clear()
            {
                positions.Clear();
                normals.Clear();
                tangents.Clear();  
                linearVels.Clear();
                angularVels.Clear();
            }
        }
        protected Observastion observastion = new Observastion();
        public Observastion Obs => observastion;

        public string skeletonFile;

        public abstract Transform GetRoot();

        public abstract void ResetSkeleton();
        public abstract void CreateSkeleton();

        public abstract void ConfigureJoints();
        public abstract void SetAnimationData(MotionFrameData motionFrameData, bool ignoreRootPos = false, bool ignoreRootRot = false);

        public abstract bool UpdateObs();

        public abstract bool HasSkeleton();

        public abstract List<Transform> GetJoints();

    }
}