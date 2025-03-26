using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using JetBrains.Annotations;
using UnityEditor;


namespace AMP
{
    public class DeepMinicSkeleton : Skeleton
    {

        public float jointFriction = 10f;
        public float lengthScale = 1.5f;
        
        public Material baseMat;

        private DeepMimicParser parser = new DeepMimicParser();
        public Transform[] jointTransforms;
        public Transform[] bodyTransforms;

        private MotionFrameData curMotionFrameData = null;


        public int numOfDofs = 0;

        public int NumOfJoints
        {
            get 
            {
                if (jointTransforms == null)
                    return 0;

                return jointTransforms.Length; 
            }
        }


        #region Create and set skeleton
        public override void ResetSkeleton()
        {
            jointTransforms = null;

            if (transform.childCount > 0)
            {
                for (int i = 0; i < transform.childCount; i++)
                {
#if UNITY_EDITOR
                    if (!Application.isPlaying)
                    {
                        DestroyImmediate(transform.GetChild(i).gameObject);
                    }
                    else
#endif
                    {
                        Destroy(transform.GetChild(i).gameObject);
                    }
                }
            }
            numOfDofs = 0;
        }

        public override void CreateSkeleton()
        {
            ResetSkeleton();

            parser.Parse(skeletonFile, false);

            // Set joints
            jointTransforms = new Transform[parser.joints.Count];
            foreach (var entry in parser.joints)
            {
                DeepMimicParser.Joint joint = entry.Value;

                GameObject jointObj = new GameObject(joint.name + " (joint)");
                Transform parent = transform;
                if(joint.parentId != -1)
                    parent = jointTransforms[joint.parentId];
                jointObj.transform.SetParent(parent);
                jointObj.transform.localPosition = joint.attachPos * lengthScale;

                jointTransforms[joint.id] = jointObj.transform ;

            }

            // Draw body shape
            Dictionary<Color, Material> colors = new Dictionary<Color, Material>();
            bodyTransforms = new Transform[parser.draws.Count];
            foreach (var entry in parser.draws)
            {
                DeepMimicParser.DrawShape body = entry.Value;

                GameObject obj = CreateBodyShapeObject(body);

                if(!colors.TryGetValue(body.color, out Material targetMat))
                {
                    targetMat = new Material(baseMat);
                    targetMat.color = body.color;
                    colors[body.color] = targetMat;
                }
                obj.GetComponent<MeshRenderer>().material = targetMat;
                bodyTransforms[body.id] = obj.transform;
            }

            UpdateObs();
        }

        public override void ConfigureJoints()
        {
            foreach (var entry in parser.joints)
            {
                DeepMimicParser.Joint joint = entry.Value;

                ConfigureJoints(jointTransforms[joint.id].gameObject, joint, parser.bodys[joint.id]);
            }

        }

        private GameObject CreateBodyShapeObject(DeepMimicParser.DrawShape body)
        {
            GameObject bodyObj = null;
            Vector3 bodyParam = body.param * lengthScale;
            if(body.shape == "sphere")
            {
                bodyObj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                SphereCollider col = bodyObj.GetComponent<SphereCollider>();
                col.radius *= 0.95f;
            }
            else if(body.shape == "capsule")
            {
                bodyObj = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                bodyParam.y -= bodyParam.y * 0.25f;
                CapsuleCollider col = bodyObj.GetComponent<CapsuleCollider>();
                col.height = 1.6f;
                col.radius = 0.45f;
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
            bodyObj.transform.localPosition = body.attachPos * lengthScale;
            bodyObj.transform.localRotation = Quaternion.Euler(body.attachTheta * Mathf.Rad2Deg);
            bodyObj.transform.localScale = bodyParam;

            return bodyObj;
        }

        //private void ConfigureJoints(GameObject jointObj, DeepMimicParser.Joint joint, DeepMimicParser.BodyShape body)
        //{
        //    Transform parent = transform;
        //    if (joint.parentId != -1)
        //        parent = jointTransforms[joint.parentId];

        //    jointObj.transform.SetParent(parent);
        //    jointObj.transform.localPosition = joint.attachPos * lengthScale;

        //    if (joint.id != 0)
        //    {
        //        ConfigurableJoint cj = jointObj.AddComponent<ConfigurableJoint>();
        //        cj.connectedBody = parent.GetComponent<Rigidbody>();
        //        cj.xMotion = ConfigurableJointMotion.Locked;
        //        cj.yMotion = ConfigurableJointMotion.Locked;
        //        cj.zMotion = ConfigurableJointMotion.Locked;
        //        cj.axis = Vector3.forward;
        //        cj.rotationDriveMode = RotationDriveMode.Slerp;

        //        if (joint.type == "spherical")
        //        {
        //            // https://github.com/xbpeng/DeepMimic/issues/146#issuecomment-807446006
        //            cj.angularXMotion = ConfigurableJointMotion.Limited;
        //            cj.angularYMotion = ConfigurableJointMotion.Limited;
        //            cj.angularZMotion = ConfigurableJointMotion.Limited;


        //            cj.lowAngularXLimit = new SoftJointLimit() { limit = joint.limLow.x * Mathf.Rad2Deg };
        //            cj.highAngularXLimit = new SoftJointLimit() { limit = joint.limHigh.x * Mathf.Rad2Deg  };


        //            numOfDofs += 3;
        //        }
        //        else if (joint.type == "revolute")
        //        {
        //            cj.angularXMotion = ConfigurableJointMotion.Limited;
        //            cj.angularYMotion = ConfigurableJointMotion.Locked;
        //            cj.angularZMotion = ConfigurableJointMotion.Locked;

        //            if (joint.limLow.x == 0)
        //            {
        //                cj.axis = -Vector3.forward;
        //            }

        //            cj.lowAngularXLimit = new SoftJointLimit() { limit = joint.limLow.x * Mathf.Rad2Deg};
        //            cj.highAngularXLimit = new SoftJointLimit() { limit = joint.limHigh.x * Mathf.Rad2Deg };

        //            numOfDofs += 1;
        //        }
        //        else
        //        {
        //            cj.angularXMotion = ConfigurableJointMotion.Locked;
        //            cj.angularYMotion = ConfigurableJointMotion.Locked;
        //            cj.angularZMotion = ConfigurableJointMotion.Locked;
        //        }
        //    }
        //    else
        //    {
        //        jointObj.AddComponent<Rigidbody>();
        //    }

        //    var rigid = jointObj.GetComponent<Rigidbody>();
        //    rigid.mass = body.mass;
        //}

        private void ConfigureJoints(GameObject jointObj, DeepMimicParser.Joint joint, DeepMimicParser.BodyShape body)
        {

            if (jointObj.name.Contains("shoulder"))
            {
                jointObj.transform.localEulerAngles += Vector3.right * 90 * -Mathf.Sign(jointObj.transform.localPosition.z);
            }

            ArticulationBody ab = jointObj.GetComponent<ArticulationBody>();
            if (ab == null)
                ab = jointObj.AddComponent<ArticulationBody>();
            ab.mass = body.mass;
            ab.jointFriction = jointFriction;

            Vector3 euler = Utils.NormalizeAngle(jointObj.transform.localEulerAngles);

            if (joint.type == "spherical")
            {
                // https://github.com/xbpeng/DeepMimic/issues/146#issuecomment-807446006
                ab.anchorRotation = Quaternion.Euler(0, 0, 0);

                ArticulationDrive xDrive = ab.xDrive;
                xDrive.driveType = ArticulationDriveType.Target;
                xDrive.lowerLimit = joint.limLow.x * Mathf.Rad2Deg- euler.x;
                xDrive.upperLimit = joint.limHigh.x * Mathf.Rad2Deg - euler.x;
                xDrive.forceLimit = joint.torqueLim;
                ab.xDrive = xDrive;

                ArticulationDrive yDrive = ab.yDrive;
                yDrive.driveType = ArticulationDriveType.Target;
                yDrive.lowerLimit = joint.limLow.y * Mathf.Rad2Deg - euler.y;
                yDrive.upperLimit = joint.limHigh.y * Mathf.Rad2Deg - euler.y;
                yDrive.forceLimit = joint.torqueLim;
                ab.yDrive = yDrive;

                ArticulationDrive zDrive = ab.zDrive;
                zDrive.driveType = ArticulationDriveType.Target;
                zDrive.lowerLimit = joint.limLow.z * Mathf.Rad2Deg - euler.z;
                zDrive.upperLimit = joint.limHigh.z * Mathf.Rad2Deg - euler.z;
                zDrive.forceLimit = joint.torqueLim;
                ab.zDrive = zDrive;

                ab.jointType = ArticulationJointType.SphericalJoint;
                ab.swingZLock = ArticulationDofLock.LimitedMotion;
                ab.swingYLock = ArticulationDofLock.LimitedMotion;
                ab.twistLock = ArticulationDofLock.LimitedMotion;

                numOfDofs += 3;
            }
            else if (joint.type == "revolute")
            {
                ab.jointType = ArticulationJointType.RevoluteJoint;
                ab.twistLock = ArticulationDofLock.LimitedMotion;
                ab.anchorRotation = Quaternion.Euler(0, 270, 0);

                ArticulationDrive xDrive = ab.xDrive;
                xDrive.driveType = ArticulationDriveType.Target;
                xDrive.lowerLimit = joint.limLow.x * Mathf.Rad2Deg - euler.z;
                xDrive.upperLimit = joint.limHigh.x * Mathf.Rad2Deg - euler.z;
                xDrive.forceLimit = joint.torqueLim;
                ab.xDrive = xDrive;

                numOfDofs += 1;
            }
            else
            {
                ab.jointType = ArticulationJointType.FixedJoint;
            }
        }


        public override void SetAnimationData(MotionFrameData motionFrameData, bool ignoreRootPos=false, bool ignoreRootRot=false)
        {
            curMotionFrameData = motionFrameData;

            parser.Parse(skeletonFile, false);

            foreach (var entry in motionFrameData.JointData)
            {
                int key = entry.Key;
                List<float> values = entry.Value;
                Transform joint = jointTransforms[key];

                if (values.Count == 4)
                {
                    joint.localRotation = new Quaternion(values[1], values[2], values[3], values[0]);

                }
                else if(values.Count == 1)
                {
                    joint.localRotation = Quaternion.Euler(0, 0, values[0] * Mathf.Rad2Deg);
                }
                else
                {
                    if(!ignoreRootPos)
                        joint.localPosition = new Vector3(values[0], values[1], values[2]) * lengthScale;
                    if(!ignoreRootRot)
                        joint.localRotation = new Quaternion(values[4], values[5], values[6], values[3]);
                }

            }
        }
        #endregion

        #region Record physics states

        public override bool UpdateObs()
        {
            observastion.Clear();
            bool valid = true;

            Transform root = jointTransforms[0];

            for (int key = 0; key < jointTransforms.Length; key++)
            {
                Transform joint = jointTransforms[key];
                if (!Utils.VectorValidate(joint.position))
                    valid = false;

                observastion.positions.Add(root.InverseTransformPoint(joint.position));

                Vector3 normal = -joint.up;
                Vector3 tangent = joint.right;
                observastion.normals.Add(normal);
                observastion.tangents.Add(tangent);

                //ArticulationBody ab = joint.GetComponent<ArticulationBody>();
                Rigidbody ab =joint.GetComponent<Rigidbody>();

                if (ab)
                {
                    // Calc linear velocity
                    if(!Utils.VectorValidate(ab.linearVelocity))
                    {
                        observastion.linearVels.Add(Vector3.zero);
                        valid = false;
                    }
                    else
                        observastion.linearVels.Add(root.InverseTransformDirection(ab.linearVelocity));

                    // Calc angular velocity
                    if (!Utils.VectorValidate(ab.angularVelocity))
                    {
                        observastion.angularVels.Add(Vector3.zero);
                        valid = false;
                    }
                    else
                        observastion.angularVels.Add(root.InverseTransformDirection(ab.angularVelocity));
                }
                else
                {
                    // Calc linear velocity
                    observastion.linearVels.Add(Vector3.zero);

                    // Calc angular velocity
                    observastion.angularVels.Add(Vector3.zero);
                }
                //else
                //{
                //    // Calc linear velocity
                //    Vector3 linearVel = (joint.position - prevState.pos) / Time.fixedDeltaTime;
                //    //observastion.linearVels.Add(linearVel);

                //    // Calc angular velocity
                //    Quaternion deltaRot = joint.rotation * Quaternion.Inverse(prevState.rot);

                //    float angle;
                //    Vector3 axis;
                //    deltaRot.ToAngleAxis(out angle, out axis);

                //    Vector3 angularVel = axis * angle * Mathf.Deg2Rad / Time.fixedDeltaTime;
                //    //observastion.angularVels.Add(angularVel);
                //    if (ab)
                //        Debug.Log($"[{ab.name}] {ab.linearVelocity}, {linearVel} || {ab.angularVelocity}, {angularVel}");
                //}
            }

            return valid;
        }
        #endregion

        #region Getter and Setter

        public override Transform GetRoot()
        {
            if (!HasSkeleton())
                return null;
            return jointTransforms[0];
        }

        public override bool HasSkeleton()
        {
            return jointTransforms != null && jointTransforms.Length > 0;
        }
        public override List<Transform> GetJoints()
        {
            return jointTransforms.ToList();
        }

        public override List<Transform> GetBodys()
        {
            return bodyTransforms.ToList();
        }

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
        #endregion

#if UNITY_EDITOR

        private void OnDrawGizmosSelected()
        {
            if (!HasSkeleton())
                return;

            parser.Parse(skeletonFile, false);
            if ( jointTransforms[0] != null)
            {
                Transform root = jointTransforms[0];
                Vector3 d = root.rotation * Vector3.right;
                d.y = 0;

                Debug.DrawLine(root.position, root.position + d, Color.magenta);
            }

            if (curMotionFrameData != null)
            {
                Dictionary<int, Vector3> parentPos = new Dictionary<int, Vector3>();
                parentPos[-1] = Vector3.zero;
                foreach(var kv in curMotionFrameData.JointData)
                {
                    int k = kv.Key;
                    if (k < 0)
                        continue;

                    // Using unity system
                    Transform joint = jointTransforms[k];
                    Vector3 normal = -joint.up;
                    Vector3 tangent = joint.right;// Vector3.Cross(normal, joint.forward).normalized;

                    Debug.DrawLine(joint.position, joint.position + normal * 0.3f, Color.green);
                    Debug.DrawLine(joint.position, joint.position + tangent * 0.3f, Color.green);


                    // from data
                    Quaternion rot = Quaternion.identity;
                    int p = k;
                    while (p > 0)
                    {
                        var values = curMotionFrameData.JointData[p];
                        rot = GetRotFromMotionData(values) * rot;
                        p = parser.joints[p].parentId;
                    }

                    normal = rot * Vector3.down;
                    tangent = rot * Vector3.right;
                    Debug.DrawLine(joint.position, joint.position + normal * .2f, Color.yellow);
                    Debug.DrawLine(joint.position, joint.position + tangent * 0.2f, Color.yellow);

                    Vector3 pp = parentPos[parser.joints[k].parentId];                    
                    Quaternion pr = Quaternion.Inverse(GetRotFromMotionData(kv.Value)) * rot;
                    Vector3 pos = pp + pr * parser.joints[k].attachPos;
                    Gizmos.color = Color.yellow;
                    Gizmos.DrawSphere(joint.position, 0.07f);
                    parentPos[k] = pos;


                    // Using unity system from scratch
                    rot = Quaternion.identity;
                    p = k;
                    while (p > 0)
                    {
                        rot = jointTransforms[p].localRotation * rot;
                        p = parser.joints[p].parentId;
                    }
                    normal = rot * Vector3.down;
                    tangent = rot * Vector3.right;

                    Debug.DrawLine(joint.position, joint.position + normal * .1f, Color.red);
                    Debug.DrawLine(joint.position, joint.position + tangent * 0.1f, Color.red);
                    // Gizmos.color = Color.red;
                    // Gizmos.DrawSphere(joint.position, 0.07f);
                }
            }
            else
            {
                if (jointTransforms[0] != null)
                {
                    Transform root = jointTransforms[0];
                    for (int key = 0; key < jointTransforms.Length; key++)
                    {
                        Transform joint = jointTransforms[key];
                        Vector3 normal = joint.up;
                        Vector3 tangent = joint.right;

                        Debug.DrawLine(joint.position, joint.position + normal * .3f, Color.green);
                        Debug.DrawLine(joint.position, joint.position + tangent * 0.3f, Color.green);


                        Quaternion rot = Quaternion.identity;
                        int p = key;
                        while (p != -1)
                        {

                            rot = jointTransforms[p].localRotation * rot;
                            p = parser.joints[p].parentId;
                        }

                        normal = rot * Vector3.down;
                        tangent = rot * Vector3.right;

                        Debug.DrawLine(joint.position, joint.position + normal * .2f, Color.red);
                        Debug.DrawLine(joint.position, joint.position + tangent * 0.2f, Color.red);
                    }
                }
            }

        }
#endif
    }

}