using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using AMP;

public class HumanAgent : Agent
{

    public Skeleton skeleton;
    public TrainingEnv env;
    public MotionDatabase motionDatabase;

    private Transform root;
    private JointDriveController controller;

    public int[] doneByContactJointIds;

    #region MLAgents function
    public override void Initialize()
    {
        if(motionDatabase == null)
            motionDatabase = FindFirstObjectByType<MotionDatabase>();

        motionDatabase.LoadDataset(false);


        // skeleton.CreateSkeleton();
        // skeleton.ConfigureJoints(); 
        
        root = skeleton.GetJoints()[0];
        controller = GetComponent<JointDriveController>();

        foreach (var joint in skeleton.GetJoints())
        {
            controller.SetupBodyPart(joint);
        }

        var joints = skeleton.GetJoints();
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
        foreach (var bodyPart in controller.bodyPartsList)
        {
            bodyPart.Reset(bodyPart);
        }
        skeleton.SetAnimationData(motionDatabase.GetRandomMotionData(), true);
        //Random start rotation to help generalize
        root.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);

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

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var continuousAct = actionBuffers.ContinuousActions;
        var i = -1;

        for (int idx = 1; idx < controller.bodyPartsList.Count; idx++)
        {
            // Spherical하고 Revoluate 관절 구분하기.
            var bodyPart = controller.bodyPartsList[idx];
            var ab = bodyPart.joint;
            if (ab)
            {
                float x = 0, y = 0, z = 0;
                if (ab.angularXMotion != ConfigurableJointMotion.Locked)
                    x = continuousAct[++i];
                if (ab.angularYMotion != ConfigurableJointMotion.Locked)
                    y = continuousAct[++i];
                if (ab.angularZMotion != ConfigurableJointMotion.Locked)
                    z = continuousAct[++i];
                bodyPart.SetJointTargetFromRotVector(x, y, z);
            }
            else
            {
                Debug.LogWarning($"Wrong joint: {bodyPart.joint}");
            }
        }
    }

    //public override void OnActionReceived(ActionBuffers actionBuffers)
    //{
    //    var continuousAct = actionBuffers.ContinuousActions;
    //    var i = -1;

    //    for(int idx = 1; idx < abController.bodyPartsList.Count; idx++)
    //    {
    //        // Spherical하고 Revoluate 관절 구분하기.
    //        var bodyPart = abController.bodyPartsList[idx];
    //        var ab = bodyPart.ab;
    //        if (ab)
    //        {
    //            if (ab.isRoot)
    //                continue;

    //            List<float> f = new List<float>();
    //            if (ab.jointType != ArticulationJointType.FixedJoint)
    //            {
    //                f.Add(continuousAct[++i]);
    //                if (ab.jointType == ArticulationJointType.SphericalJoint)
    //                {
    //                    f.Add(continuousAct[++i]);
    //                    f.Add(continuousAct[++i]);
    //                }
    //                bodyPart.SetJointTargetFromRotVector(f);
    //            }
    //        }
    //        else
    //        {
    //            Debug.LogWarning($"Wrong joint: {bodyPart.ab}");
    //        }
    //    }
    //}

    private void FixedUpdate()
    {
        int status = IsNotNormal();
        if (status != 0)
        {
            Debug.Log($"{transform.parent.name} End episode {status}");
            EndEpisode();
            return;
        }
        if (skeleton.HasSkeleton())
        {
            skeleton.UpdateObs();
        }

        if (controller.bodyPartsList.Count > 0)
        {
            AddReward(env.GetReward(controller));
        }
    }

    private int IsNotNormal()
    {
        if (float.IsNaN(transform.position.x) || float.IsNaN(transform.position.y) || float.IsNaN(transform.position.z) ||
            float.IsInfinity(transform.position.x) || float.IsInfinity(transform.position.y) || float.IsInfinity(transform.position.z))
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
