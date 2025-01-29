using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using AMP;
using Unity.VisualScripting;

public class HumanAgent : Agent
{
    public Direction cmvDir;

    public Skeleton skeleton;
    public TrainingEnv env;

    private MotionDatabase motionDatabase;
    private Transform root;
    private ArticulationBodyController controller;

    public int[] doneByContactJointIds;

    private Vector3 initPos;

    #region MLAgents function
    public override void Initialize()
    {
        if(motionDatabase == null)
            motionDatabase = FindFirstObjectByType<MotionDatabase>();

        motionDatabase.LoadDataset(false);
        controller = GetComponent<ArticulationBodyController>();
        SetupSkeleton();
    }

    private void SetupSkeleton()
    {
        controller.ResetState();
        skeleton.CreateSkeleton();
        skeleton.ConfigureJoints();

        root = skeleton.GetJoints()[0];
        initPos = root.position;

        var joints = skeleton.GetJoints();
        for (int key = 0; key < joints.Count; key++)
        {
            controller.SetupBodyPart(key, joints[key]);
        }

        for (int i = 0; i < doneByContactJointIds.Length; i++)
        {
            var t = joints[doneByContactJointIds[i]].GetComponent<GroundContact>();

            t.agentDoneOnGroundContact = true;
            t.penalizeGroundContact = true;
            t.groundContactPenalty = -1;
        }
    }

    public override void OnEpisodeBegin()
    {
        if(!skeleton.HasSkeleton())
        {
            SetupSkeleton();
        }

        //Random start rotation to help generalize
        var randomRot = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);
        controller.bodyPartsDict[0].ab.TeleportRoot(initPos, randomRot);

        var motionData = motionDatabase.GetRandomMotionData();
        foreach (var ent in controller.bodyPartsDict)
        {
            var key = ent.Key;
            var bodyPart = ent.Value;
            Vector3 euler = Vector3.zero;

            if(motionData.JointData.ContainsKey(key))
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
            }
            bodyPart.Reset(euler);
        }

        skeleton.UpdateObs();
        env.BeginEpisode();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Skeleton.Observastion obs = skeleton.Obs;

        for(int i = 0; i < obs.normals.Count; i++)
        {
            sensor.AddObservation(obs.positions[i]);
            sensor.AddObservation(obs.normals[i]);
            sensor.AddObservation(obs.tangents[i]);
            sensor.AddObservation(obs.linearVels[i]);
            sensor.AddObservation(obs.angularVels[i]);
        }

        var goals = env.GetGoals(skeleton);
        foreach (var g in goals)
        {
            sensor.AddObservation(g);
        }
    }

    //public override void OnActionReceived(ActionBuffers actionBuffers)
    //{
    //    var continuousAct = actionBuffers.ContinuousActions;
    //    var i = -1;

    //    for (int idx = 1; idx < controller.bodyPartsList.Count; idx++)
    //    {
    //        // Spherical하고 Revoluate 관절 구분하기.
    //        var bodyPart = controller.bodyPartsList[idx];
    //        var ab = bodyPart.joint;
    //        if (ab)
    //        {
    //            float x = 0, y = 0, z = 0;
    //            if (ab.angularXMotion != ConfigurableJointMotion.Locked)
    //                x = continuousAct[++i];
    //            if (ab.angularYMotion != ConfigurableJointMotion.Locked)
    //                y = continuousAct[++i];
    //            if (ab.angularZMotion != ConfigurableJointMotion.Locked)
    //                z = continuousAct[++i];
    //            bodyPart.SetJointTargetFromRotVector(x, y, z);
    //        }
    //        else
    //        {
    //            Debug.LogWarning($"Wrong joint: {bodyPart.joint}");
    //        }
    //    }
    //}

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var continuousAct = actionBuffers.ContinuousActions;
        var i = -1;

        for (int idx = 1; idx < controller.bodyPartsList.Count; idx++)
        {
            // Spherical하고 Revoluate 관절 구분하기.
            var bodyPart = controller.bodyPartsList[idx];
            var ab = bodyPart.ab;
            if (ab)
            {
                if (ab.isRoot)
                    continue;

                List<float> f = new List<float>();
                if (ab.jointType != ArticulationJointType.FixedJoint)
                {
                    f.Add(continuousAct[++i]);
                    if (ab.jointType == ArticulationJointType.SphericalJoint)
                    {
                        f.Add(continuousAct[++i]);
                        f.Add(continuousAct[++i]);
                    }
                    bodyPart.SetJointTargetFromRotVector(f);
                }
            }
            else
            {
                Debug.LogWarning($"Wrong joint: {bodyPart.ab}");
            }
        }
    }

    private void Update()
    {
        if (controller == null || cmvDir == null)
            return;

        Vector3 cmv = Vector3.zero;
        float totalMass = 0;
        foreach (var bp in controller.bodyPartsList)
        {
            cmv += bp.ab.linearVelocity * bp.ab.mass;
            totalMass += bp.ab.mass;
        }
        cmv = cmv / totalMass;
        var vec = new Vector2(cmv.x, -cmv.z).normalized;

        cmvDir.SetHeading(Mathf.Atan2(vec.y, vec.x));
    }

    private void FixedUpdate()
    {
        int status = ValidateAndUpdateObs();
        if (status != 0)
        {
            Debug.Log($"{transform.parent.name} End episode {status}");
            SetupSkeleton();
            SetReward(-10f);
            EndEpisode();
            return;
        }

        if (controller.bodyPartsList.Count > 0)
        {
            var joints = skeleton.GetJoints();
            var left_ankle = joints[11];
            var right_ankle = joints[5];
            var neck = joints[2];

            var ankle_mid = (left_ankle.position + right_ankle.position) / 2;
            var ankle_to_neck = (neck.position - ankle_mid).normalized;
            if (Vector3.Angle(Vector3.up, ankle_to_neck) > 50)
            {
                SetReward(-1f);
                EndEpisode();
            }
            else
            {

                AddReward(env.GetReward(controller));
            }
        }
    }

    private int ValidateAndUpdateObs()
    {
        if (skeleton.HasSkeleton())
        {
            if (!skeleton.UpdateObs())
            {
                return 5;
            }
        }

        if (root != null && !Utils.VectorValidate(root.transform.position))
        {
            return 1;
        }

        if (Vector3.Distance(transform.position, env.transform.position) > 10000)
            return 2;

        if (controller.bodyPartsList.Count > 0)
        {
            float r = env.GetReward(controller);
            if (float.IsNaN(r) || float.IsInfinity(r))
                return 3;
        }

        if (!env.ValidateEnvironment(skeleton))
            return 4;

        return 0;
    }

    #endregion

}
