using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using JetBrains.Annotations;


namespace AMP
{
    public class DeepMinicSkeleton : Skeleton
    {
        public class JointState
        {
            public Vector3 pos;
            public Quaternion rot;
        }


        public float LENGTH_SCALE = 10.0f;
        
        public Material baseMat;

        private DeepMimicParser parser = new DeepMimicParser();

        private SortedDictionary<int, Transform> jointTransforms = new SortedDictionary<int, Transform>();
        private Dictionary<int, Transform> bodyTransforms = new Dictionary<int, Transform>();
        private Dictionary<Color, Material> colors = new Dictionary<Color, Material>();

        private Dictionary<int, JointState> prevStates = new Dictionary<int, JointState>();

        private MotionFrameData curMotionFrameData = null;

        public int numOfDofs = 0;

        public int NumOfJoints
        {
            get { return jointTransforms.Count; }
        }

        public bool HasSkeleton
        {
            get
            {
                return jointTransforms.Count > 0;
            }
        }


        #region Create and control skeleton
        public void ResetSkeleton()
        {
            colors.Clear();
            jointTransforms.Clear();
            bodyTransforms.Clear();
            prevStates.Clear();

            if (transform.childCount > 0)
            {
                for (int i = 0; i < transform.childCount; i++)
                {
                    DestroyImmediate(transform.GetChild(i).gameObject);
                }
            }
            numOfDofs = 0;
        }
        public override void RecordPrevState()
        {
            foreach(var entry in jointTransforms)
            {
                int key = entry.Key;
                Transform joint = entry.Value;

                if (prevStates.ContainsKey(key))
                {
                    prevStates[key].pos = joint.position;
                    prevStates[key].rot = joint.rotation;
                }
                else
                {
                    prevStates[key] = new JointState() { pos = joint.position, rot = joint.rotation };
                }
            }
        }

        public override void CreateSkeleton()
        {
            ResetSkeleton();

            parser.Parse(skeletonFile);

            // Set joints
            foreach (var entry in parser.joints)
            {
                DeepMimicParser.Joint joint = entry.Value;

                GameObject jointObj = CreateJointObject(joint, parser.bodys[entry.Key]);
                jointTransforms[joint.id] = jointObj.transform;

            }

            // Draw body shape
            foreach (var entry in parser.draws)
            {
                DeepMimicParser.DrawShape body = entry.Value;

                GameObject obj = CreateBodyShapeObject(body);
                bodyTransforms[entry.Key] = obj.transform;

                if(!colors.TryGetValue(body.color, out Material targetMat))
                {
                    targetMat = new Material(baseMat);
                    targetMat.color = body.color;
                    colors[body.color] = targetMat;
                }
                obj.GetComponent<MeshRenderer>().material = targetMat;
            }
            foreach (var entry in bodyTransforms)
            {
                entry.Value.localScale = parser.draws[entry.Key].param * LENGTH_SCALE;
            }

            RecordPrevState();
        }

        private GameObject CreateBodyShapeObject(DeepMimicParser.DrawShape body)
        {
            GameObject bodyObj = null;
            Vector3 mod = Vector3.zero;
            if(body.shape == "sphere")
            {
                bodyObj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            }
            else if(body.shape == "capsule")
            {
                bodyObj = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                mod.y = -1f;
            }
            else if(body.shape == "box")
            {
                bodyObj = GameObject.CreatePrimitive(PrimitiveType.Cube);
            }
            else
            {
                Debug.LogError($"Missing type implementation {body.shape}");
                return null;
            }


            // Set shapes
            bodyObj.name = body.name + " (body)";

            Transform parent = jointTransforms[body.parentId];

            bodyObj.transform.SetParent(parent);
            bodyObj.transform.localPosition = body.attachPos * LENGTH_SCALE;
            bodyObj.transform.localRotation = Quaternion.Euler(body.attachTheta * Mathf.Rad2Deg);
            bodyObj.transform.localScale = body.param * LENGTH_SCALE;

            return bodyObj;
        }

        private GameObject CreateJointObject(DeepMimicParser.Joint joint, DeepMimicParser.BodyShape body)
        {
            GameObject jointObj = new GameObject(joint.name + " (joint)");

            Transform parent = transform;
            if (jointTransforms.ContainsKey(joint.parentId))
            {
                parent = jointTransforms[joint.parentId];
            }
            jointObj.transform.SetParent(parent);
            jointObj.transform.localPosition = joint.attachPos * LENGTH_SCALE;

            if (joint.id != 0)
            {
                ConfigurableJoint cj = jointObj.AddComponent<ConfigurableJoint>();
                cj.connectedBody = parent.GetComponent<Rigidbody>();
                cj.xMotion = ConfigurableJointMotion.Locked;
                cj.yMotion = ConfigurableJointMotion.Locked;
                cj.zMotion = ConfigurableJointMotion.Locked;
                cj.axis = Vector3.forward;
                cj.rotationDriveMode = RotationDriveMode.Slerp;

                if (joint.type == "spherical")
                {
                    // https://github.com/xbpeng/DeepMimic/issues/146#issuecomment-807446006
                    cj.angularXMotion = ConfigurableJointMotion.Free;
                    cj.angularYMotion = ConfigurableJointMotion.Free;
                    cj.angularZMotion = ConfigurableJointMotion.Free;
                    numOfDofs += 3;
                }
                else if(joint.type == "revolute")
                {
                    cj.angularXMotion = ConfigurableJointMotion.Limited;
                    cj.angularYMotion = ConfigurableJointMotion.Locked;
                    cj.angularZMotion = ConfigurableJointMotion.Locked;

                    if(joint.limLow.x == 0)
                    {
                        cj.axis = -Vector3.forward;
                    }

                    cj.lowAngularXLimit = new SoftJointLimit() { limit = joint.limLow.x * Mathf.Rad2Deg +cj.axis.z * 90};
                    cj.highAngularXLimit = new SoftJointLimit() { limit = joint.limHigh.x * Mathf.Rad2Deg + cj.axis.z * 90 };

                    numOfDofs += 1;
                }
                else
                {
                    cj.angularXMotion = ConfigurableJointMotion.Locked;
                    cj.angularYMotion = ConfigurableJointMotion.Locked;
                    cj.angularZMotion = ConfigurableJointMotion.Locked;
                }
            }
            else
            {
                jointObj.AddComponent<Rigidbody>();
            }

            var rigid = jointObj.GetComponent<Rigidbody>();
            rigid.mass = body.mass;

            return jointObj;

        }
        public override void SetAnimationData(MotionFrameData motionFrameData)
        {
            RecordPrevState();
            curMotionFrameData = motionFrameData;

            foreach (var entry in motionFrameData.JointData)
            {
                int key = int.Parse(entry.Key);
                List<float> values = entry.Value;
                Transform joint = jointTransforms[key];

                if(values.Count == 4)
                {
                    joint.localRotation = new Quaternion(values[1], values[2], values[3], values[0]);
                }
                else if(values.Count == 1)
                {
                    joint.localRotation = Quaternion.Euler(0, 0, values[0] * Mathf.Rad2Deg);
                }
                else
                {
                    joint.localPosition = new Vector3(values[0], values[1], values[2]) * LENGTH_SCALE;
                    joint.localRotation = new Quaternion(values[4], values[5], values[6], values[3]);
                }
            }
            GetObs();
        }


        public override Observastion GetObs()
        {
            Observastion obs = new Observastion();

            Transform root = jointTransforms[0];

            foreach (var entry in jointTransforms)
            {
                int key = entry.Key;
                Transform joint = entry.Value;
                JointState prevState = prevStates[key];

                obs.positions.Add(root.InverseTransformPoint(joint.position));

                obs.normals.Add(joint.right);
                obs.tangents.Add(Vector3.Cross(joint.right, Vector3.up));

                // Calc linear velocity
                Vector3 linearVel = (joint.position - prevState.pos) / Time.deltaTime;
                obs.linearVels.Add(linearVel);

                // Calc angular velocity
                Quaternion deltaRot = joint.rotation * Quaternion.Inverse(prevState.rot);

                float angle;
                Vector3 axis;
                deltaRot.ToAngleAxis(out angle, out axis);

                Vector3 angularVel = axis * angle * Mathf.Deg2Rad / Time.deltaTime;
                obs.angularVels.Add(angularVel);
            }

            return obs;
        }

        public override List<Transform> GetJoints()
        {
            List<Transform> joints = new List<Transform> ();

            foreach(var joint in jointTransforms)
            {
                joints.Add(joint.Value);
            }

            return joints;
        }

        public override Transform GetBody(int id)
        {
            return bodyTransforms[id];
        }
        #endregion


        private Quaternion GetRotFromMotionData(List<float> values)
        {
            if (values.Count == 4)
            {
                return new Quaternion(values[1], values[2], values[3], values[0]);
            }
            else if (values.Count == 1)
            {
                return new Quaternion(0, 0, Mathf.Sin(values[0] / 2), Mathf.Cos(values[0] / 2));
            }
            else
            {
                return new Quaternion(values[4], values[5], values[6], values[3]);
            }
        }

        private void OnDrawGizmosSelected()
        {
            if (curMotionFrameData != null)
            {
                Dictionary<int, Vector3> parentPos = new Dictionary<int, Vector3>();
                parentPos[-1] = Vector3.zero;
                foreach(var kv in curMotionFrameData.JointData)
                {
                    int k = int.Parse(kv.Key);
                    if (k < 0)
                        continue;

                    // Using unity system
                    Transform joint = jointTransforms[k];
                    Vector3 normal = joint.right;
                    Vector3 tangent = joint.forward;

                    Debug.DrawLine(joint.position, joint.position + normal, Color.green);


                    // from data
                    Quaternion rot = Quaternion.identity;
                    int p = k;
                    while (p != -1)
                    {
                        var values = curMotionFrameData.JointData[p + ""];
                        rot = GetRotFromMotionData(values) * rot;
                        p = parser.joints[p].parentId;
                    }

                    normal = rot * Vector3.right;
                    Debug.DrawLine(joint.position, joint.position + normal * (2 / 3f), Color.yellow);

                    Vector3 pp = parentPos[parser.joints[k].parentId];                    
                    Quaternion pr = Quaternion.Inverse(GetRotFromMotionData(kv.Value)) * rot;
                    Vector3 pos = pp + pr * parser.joints[k].attachPos;
                    Gizmos.color = Color.yellow;
                    Gizmos.DrawSphere(joint.position, 0.07f);
                    parentPos[k] = pos;


                    // Using unity system from scratch
                    rot = Quaternion.identity;
                    p = k;
                    while (p != -1)
                    {
                        rot = jointTransforms[p].localRotation * rot;
                        p = parser.joints[p].parentId;
                    }
                    normal = rot * Vector3.right;

                    Debug.DrawLine(joint.position, joint.position + normal * (1 / 3f), Color.red);
                    // Gizmos.color = Color.red;
                    // Gizmos.DrawSphere(joint.position, 0.07f);
                }
            }
            else
            {

                foreach (var kv in jointTransforms)
                {
                    Transform joint = kv.Value;
                    Vector3 normal = joint.right;
                    Vector3 tangent = joint.forward;

                    Debug.DrawLine(joint.position, joint.position + normal, Color.green);


                    Quaternion rot = Quaternion.identity;
                    int p = kv.Key;
                    while (p != -1)
                    {
                        rot = jointTransforms[p].localRotation * rot;
                        p = parser.joints[p].parentId;
                    }

                    normal = rot * Vector3.right;
                    tangent = rot * Vector3.forward;

                    Debug.DrawLine(joint.position, joint.position + normal * 2/ 3f, Color.red);
                }
            }

        }
    }

}