using System.Collections;
using System.Collections.Generic;
using UnityEngine;

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
        }

        public UnityEngine.Object skeletonFile;

        public abstract void CreateSkeleton();

        public abstract void SetAnimationData(MotionFrameData motionFrameData);

        public abstract void RecordPrevState();

        public abstract Observastion GetObs();

        public abstract List<Transform> GetJoints();

        public abstract Transform GetBody(int id);
    }
}