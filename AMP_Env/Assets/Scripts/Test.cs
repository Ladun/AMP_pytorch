using AMP;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using UnityEngine;
using static UnityEngine.GraphicsBuffer;

public class Test : MonoBehaviour
{
    public DeepMinicSkeleton skeleton;
    public DeepMinicSkeleton testSkeleton;
    public MotionDatabase motionDatabase;

    public Direction testDir;
    public Vector2 testVec;

    private ArticulationBodyController controller;
    private Vector3 initPos;


    private void Awake()
    {
        if(testSkeleton)
            testSkeleton.CreateSkeleton();


        skeleton.CreateSkeleton();
        skeleton.ConfigureJoints();
        var joints = skeleton.GetJoints();
        var root = joints[0];

        root.GetComponent<ArticulationBody>().immovable = true;

        controller = skeleton.transform.GetComponent<ArticulationBodyController>();
        for (int key = 0; key < joints.Count; key++)
        {
            controller.SetupBodyPart(key, joints[key]);
        }
        initPos = root.position;
    }


    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.T))
        {
            if (Time.timeScale > 0.1f)
                Time.timeScale -= 0.1f;
            else if (Time.timeScale > 0.01f)
                Time.timeScale -= 0.01f;
        }

        if (Input.GetKeyDown(KeyCode.Alpha1))
            Time.timeScale = 1;
        if (Input.GetKeyDown(KeyCode.Alpha2))
            Time.timeScale *= 0.5f;
        if (Input.GetKeyDown(KeyCode.Alpha3))
            Time.timeScale = 20;

        testDir.SetHeading(Mathf.Atan2(testVec.y, testVec.x));
        ControlArticulationBody();
    }


    private void ControlConfigurableJoint()
    {  // Configurable Joint Test
        if (Input.GetKeyDown(KeyCode.U))
        {
            var jointTransforms = skeleton.GetJoints();
            foreach (var joint in jointTransforms)
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

            foreach (var motion in motionData.JointData)
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
                        euler.x = values[0] * Mathf.Rad2Deg;
                    }

                    //euler.x = Mathf.Lerp(joint.lowAngularXLimit.limit, joint.highAngularXLimit.limit, euler.x);
                    //euler.y = Mathf.Lerp(-joint.angularYLimit.limit, joint.angularYLimit.limit, euler.y);
                    //euler.z = Mathf.Lerp(-joint.angularZLimit.limit, joint.angularZLimit.limit, euler.z);


                    //joint.targetRotation = Quaternion.Euler(euler);
                    joint.SetTargetRotationLocal(Quaternion.Euler(euler), Quaternion.identity);

                }


            }
        }

    }
    

    private void ControlArticulationBody()
    {
        if (Input.GetKeyDown(KeyCode.U))
        {
            var jointTransforms = skeleton.GetJoints();
            foreach (var joint in jointTransforms)
            {
                ArticulationBody ab = joint.GetComponent<ArticulationBody>();
                if (ab)
                {
                    ab.SetDriveTarget(ArticulationDriveAxis.X, 0);
                    ab.SetDriveTarget(ArticulationDriveAxis.Y, 0);
                    ab.SetDriveTarget(ArticulationDriveAxis.Z, 0);

                    if (ab.jointType == ArticulationJointType.SphericalJoint)
                    {
                        ab.resetJointPosition(Vector3.zero);
                    }
                    else if (ab.jointType == ArticulationJointType.RevoluteJoint)
                    {
                        ab.resetJointPosition(0);
                    }
                }
            }
        }

        // Articulation Body Test
        if (Input.GetKeyDown(KeyCode.I))
        {
            if (!skeleton.HasSkeleton())
            {
                skeleton.CreateSkeleton();
                skeleton.ConfigureJoints();
                Debug.Log("Create new skeleton");
            }

            if (!motionDatabase.HasMotion)
                motionDatabase.LoadDataset();

            var motionData = motionDatabase.GetRandomMotionData();
            testSkeleton.SetAnimationData(motionData, true, true);

            var jointTransforms = skeleton.GetJoints();

            if (Input.GetKey(KeyCode.L))
            {
                var randomRot = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);
                controller.bodyPartsDict[0].ab.TeleportRoot(initPos, randomRot);
                foreach (var ent in controller.bodyPartsDict)
                {
                    var key = ent.Key;
                    var bodyPart = ent.Value;
                    Vector3 euler = Vector3.zero;

                    if (motionData.JointData.ContainsKey(key))
                    {
                        var values = motionData.JointData[key];

                        if (values.Count == 4)
                        {
                            euler = Utils.NormalizeAngle(new Quaternion(values[1], values[2], values[3], values[0]).eulerAngles);
                        }
                        else if (values.Count == 1)
                        {
                            euler = Utils.NormalizeAngle(Quaternion.Euler(0, 0, values[0] * Mathf.Rad2Deg).eulerAngles);
                        }
                        bodyPart.Reset(euler);
                    }
                }
            }
            else
            {
                foreach (var motion in motionData.JointData)
                {
                    ArticulationBody ab = jointTransforms[motion.Key].GetComponent<ArticulationBody>();
                    List<float> values = motion.Value;
                    Vector3 euler = Vector3.zero;

                    if (ab.isRoot)
                    {
                        if (ab.isRoot && !ab.immovable)
                        {
                            Vector3 pos = new Vector3(values[0], values[1], values[2]) * skeleton.lengthScale;
                            euler = Utils.NormalizeAngle(new Quaternion(values[4], values[5], values[6], values[3]).eulerAngles);
                            ab.TeleportRoot(pos, Quaternion.Euler(euler));
                        }
                    }
                    else
                    {
                        if (values.Count == 4)
                        {
                            euler = Utils.NormalizeAngle(new Quaternion(values[1], values[2], values[3], values[0]).eulerAngles);
                            
                        }
                        else if (values.Count == 1)
                        {
                            euler = Utils.NormalizeAngle(Quaternion.Euler(0, 0, values[0] * Mathf.Rad2Deg).eulerAngles);
                        }

                        if (Input.GetKey(KeyCode.J))
                        {
                            if (ab.jointType == ArticulationJointType.SphericalJoint)
                            {
                                ab.SetDriveRotation(Quaternion.Euler(euler));
                            }
                            else if (ab.jointType == ArticulationJointType.RevoluteJoint)
                            {
                                ab.SetDriveTarget(ArticulationDriveAxis.X, euler.z);
                            }
                        }
                        if (Input.GetKey(KeyCode.K))
                        {
                            if (ab.jointType == ArticulationJointType.SphericalJoint)
                            {
                                ab.resetJointPosition(ab.ToTargetRotationInReducedSpace(Quaternion.Euler(euler), false));
                            }
                            else if (ab.jointType == ArticulationJointType.RevoluteJoint)
                            {
                                ab.resetJointPosition(euler.z * Mathf.Deg2Rad);
                            }
                        }
                    }
                }
            }

        }
    }
}
