using AMP;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using UnityEngine;

public class Test : MonoBehaviour
{
    public DeepMinicSkeleton skeleton;
    public MotionDatabase motionDatabase;

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.T))
        {
            if(Time.timeScale > 0.1f)
                Time.timeScale -= 0.1f;
            else if(Time.timeScale > 0.01f)
                Time.timeScale -= 0.01f;
        }
        if (Input.GetKeyDown(KeyCode.Y))
            Time.timeScale = 1f;

        if (Input.GetKeyDown(KeyCode.U))
        {
            var jointTransforms = skeleton.GetJoints();
            foreach(Transform joint in jointTransforms)
            {
                ConfigurableJoint cj = joint.GetComponent<ConfigurableJoint>();
                if (cj)
                    cj.targetRotation = Quaternion.identity;
            }
        }
            
        if (Input.GetKeyDown(KeyCode.I))
        {
            if (!motionDatabase.HasMotion)
                motionDatabase.LoadDataset();

            var jointTransforms = skeleton.GetJoints();
            //var motionData = motionDatabase.GetMotionData(motionDatabase.GetMotionKeys()[0])[0];
            var motionData = motionDatabase.GetRandomMotionData();
            skeleton.SetAnimationData(motionData, true, true);

            foreach(var motion in motionData.JointData)
            {
                int key = motion.Key;
                List<float> values = motion.Value;

                ConfigurableJoint joint = jointTransforms[key].GetComponent<ConfigurableJoint>();

                if (joint)
                {
                    Vector3 euler = Vector3.zero;
                    if (values.Count == 4)
                    {
                        euler = new Quaternion(values[1], values[2], values[3], values[0]).eulerAngles;
                    }
                    else
                    {
                        euler.x= values[0] * Mathf.Rad2Deg;
                    }

                    //euler.x = Mathf.Lerp(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit, euler.x);
                    //euler.y = Mathf.Lerp(-joint.angularYLimit.limit, joint.angularYLimit.limit, euler.y);
                    //euler.z = Mathf.Lerp(-joint.angularZLimit.limit, joint.angularZLimit.limit, euler.z);

                    
                    joint.targetRotation = Quaternion.Euler(euler);

                }


            }
        }


        if(Input.GetKeyDown(KeyCode.X))
        {
            if (!skeleton.HasSkeleton())
            {
                skeleton.CreateSkeleton();
                skeleton.ConfigureJoints();
            }

            if (!motionDatabase.HasMotion)
                motionDatabase.LoadDataset();

            var motionData = motionDatabase.GetRandomMotionData();
            var jointTransforms = skeleton.GetJoints();
            Dictionary<Transform, int> jointToKey = new Dictionary<Transform, int>();
            for (int key = 0; key < jointTransforms.Count; key++)
            {
                jointToKey[jointTransforms[key]] = key;
            }

            List<float> f = new List<float>();

            Stack<ArticulationBody> st = new Stack<ArticulationBody>();
            ArticulationBody rootAb = jointTransforms[0].GetComponent<ArticulationBody>();
            st.Push(rootAb);

            while(st.Count > 0)
            {
                ArticulationBody ab = st.Pop();
                List<float> values = motionData.JointData[jointToKey[ab.transform]];
                Vector3 euler = Vector3.zero;

                if (ab.isRoot)
                {
                    if (ab.isRoot && !ab.immovable)
                    {
                        Vector3 pos = new Vector3(values[0], values[1], values[2]) * skeleton.lengthScale;
                        euler = Utils.NormalizeAngle(new Quaternion(values[4], values[5], values[6], values[3]).eulerAngles);

                        f.Add(pos.x);
                        f.Add(pos.y);
                        f.Add(pos.z);
                        f.Add(euler.x * Mathf.Deg2Rad);
                        f.Add(euler.y * Mathf.Deg2Rad);
                        f.Add(euler.z * Mathf.Deg2Rad);

                        ab.TeleportRoot(pos, Quaternion.Euler(euler));
                    }
                }
                else
                {

                    if (values.Count == 4)
                    {
                        euler = Utils.NormalizeAngle(new Quaternion(values[1], values[2], values[3], values[0]).eulerAngles);
                        f.Add(euler.x * Mathf.Deg2Rad);
                        f.Add(euler.y * Mathf.Deg2Rad);
                        f.Add(euler.z * Mathf.Deg2Rad);
                    }
                    else if (values.Count == 1)
                    {
                        euler = Utils.NormalizeAngle(Quaternion.Euler(0, 0, values[0] * Mathf.Rad2Deg).eulerAngles);
                        f.Add(euler.z * Mathf.Deg2Rad);
                    }


                    if (Input.GetKey(KeyCode.Z))
                    {
                        if (ab.jointType == ArticulationJointType.SphericalJoint)
                        {
                            ab.SetDriveTarget(ArticulationDriveAxis.X, euler.x);
                            ab.SetDriveTarget(ArticulationDriveAxis.Y, euler.y);
                            ab.SetDriveTarget(ArticulationDriveAxis.Z, euler.z);
                        }
                        else if (ab.jointType == ArticulationJointType.RevoluteJoint)
                        {
                            ab.SetDriveTarget(ArticulationDriveAxis.X, euler.z);
                        }
                    }
                }


                for (int i = 0; i < ab.transform.childCount; i++)
                {
                    ArticulationBody a = ab.transform.GetChild(i).GetComponent<ArticulationBody>();
                    if(a && (a.jointType == ArticulationJointType.SphericalJoint || a.jointType == ArticulationJointType.RevoluteJoint))
                        st.Push(a);
                }
            }

            if (Input.GetKey(KeyCode.C))
            {
                rootAb.SetJointPositions(f);
            }

            //foreach (var motion in motionData.JointData)
            //{
            //    ArticulationBody ab = jointDict[motion.Key].GetComponent<ArticulationBody>();
            //    List<float> values = motion.Value;
            //    Vector3 euler = Vector3.zero;

            //    if (ab.isRoot)
            //    {
            //        ab.immovable = true;
            //        if (ab.isRoot && !ab.immovable)
            //        {
            //            Vector3 pos = new Vector3(values[0], values[1], values[2]) * skeleton.lengthScale;
            //            euler = Utils.NormalizeAngle(new Quaternion(values[4], values[5], values[6], values[3]).eulerAngles);

            //            f.Add(pos.x);
            //            f.Add(pos.y);
            //            f.Add(pos.z);
            //            f.Add(euler.x * Mathf.Deg2Rad);
            //            f.Add(euler.y * Mathf.Deg2Rad);
            //            f.Add(euler.z * Mathf.Deg2Rad);
            //        }
            //    }
            //    else
            //    {

            //        if (values.Count == 4)
            //        {
            //            euler = Utils.NormalizeAngle(new Quaternion(values[1], values[2], values[3], values[0]).eulerAngles);
            //            f.Add(euler.x * Mathf.Deg2Rad);
            //            f.Add(euler.y * Mathf.Deg2Rad);
            //            f.Add(euler.z * Mathf.Deg2Rad);
            //        }
            //        else if (values.Count == 1)
            //        {
            //            euler = Utils.NormalizeAngle(Quaternion.Euler(0, 0, values[0] * Mathf.Rad2Deg).eulerAngles);
            //            f.Add(euler.z * Mathf.Deg2Rad);
            //        }


            //        if (ab.jointType == ArticulationJointType.SphericalJoint)
            //        {
            //            ab.SetDriveTarget(ArticulationDriveAxis.X, euler.x);
            //            ab.SetDriveTarget(ArticulationDriveAxis.Y, euler.y);
            //            ab.SetDriveTarget(ArticulationDriveAxis.Z, euler.z);
            //        }
            //        else if (ab.jointType == ArticulationJointType.RevoluteJoint)
            //        {
            //            ab.SetDriveTarget(ArticulationDriveAxis.X, euler.z);
            //        }
            //    }
            //}


        }
    }
}
