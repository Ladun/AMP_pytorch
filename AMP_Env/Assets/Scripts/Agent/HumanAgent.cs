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
    private ArticulationBodyController abController;

    public int[] doneByContactJointIds;

    #region MLAgents function
    public override void Initialize()
    {
        if(motionDatabase == null)
            motionDatabase = FindFirstObjectByType<MotionDatabase>();

        motionDatabase.LoadDataset(false);

        abController = GetComponent<ArticulationBodyController>();


    }
    public override void OnEpisodeBegin()
    {

        skeleton.CreateSkeleton();
        root = skeleton.GetJoints()[0];

        var motionData = motionDatabase.GetRandomMotionData();
        skeleton.SetAnimationData(motionDatabase.GetRandomMotionData(), true);
        //Random start rotation to help generalize
        root.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);

        skeleton.AddJoints();
        abController.ResetState();
        foreach (var e in skeleton.GetJointsDict())
        {
            abController.SetupBodyPart(e.Key, e.Value);
        }
        var joints = skeleton.GetJoints();
        for (int i = 0; i < doneByContactJointIds.Length; i++)
        {
            var t = joints[doneByContactJointIds[i]].GetComponent<GroundContact>();

            t.agentDoneOnGroundContact = true;
            t.penalizeGroundContact = true;
            t.groundContactPenalty = -1;
        }

        env.BeginEpisode();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Skeleton.Observastion obs = skeleton.GetObs();

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

        for(int idx = 1; idx < abController.bodyPartsList.Count; idx++)
        {
            // Spherical하고 Revoluate 관절 구분하기.
            var bodyPart = abController.bodyPartsList[idx];
            var ab = bodyPart.ab;
            if (ab)
            {
                if (ab.isRoot)
                    continue;

                float x = 0, y = 0, z = 0;
                if (ab.jointType != ArticulationJointType.FixedJoint)
                {
                    x = continuousAct[++i];
                    if (ab.jointType == ArticulationJointType.SphericalJoint)
                    {
                        y = continuousAct[++i];
                        z = continuousAct[++i];
                    }
                    bodyPart.SetJointTargetFromRotVector(x, y, z);
                }
            }
            else
            {
                Debug.LogWarning($"Wrong joint: {bodyPart.ab}");
            }
        }
    }

    private void FixedUpdate()
    {
        skeleton.RecordPrevState();

        if(abController.bodyPartsList.Count > 0)
            AddReward(env.GetReward(abController));
    }

    #endregion

}
