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

    public Skeleton skeleton;
    public TrainingEnv env;

    private Transform root;
    private JointDriveController jdController;

    public int[] doneByContactJointIds;

    #region MLAgents function
    public override void Initialize()
    {
        skeleton.CreateSkeleton();
        root = skeleton.GetJoints()[0];
        jdController = GetComponent<JointDriveController>();

        foreach(var joint in skeleton.GetJoints())
        {
            jdController.SetupBodyPart(joint);
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
        // Reset all joint
        foreach (var bodyPart in jdController.bodyPartsList)
        {
            bodyPart.Reset(bodyPart);
        }

        //Random start rotation to help generalize
        root.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);

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
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var continuousAct = actionBuffers.ContinuousActions;
        var i = -1;

        for(int idx = 1; idx < jdController.bodyPartsList.Count; idx++)
        {
            // Spherical하고 Revoluate 관절 구분하기.
            var bodyPart = jdController.bodyPartsList[idx];
            var joint = bodyPart.rb.GetComponent<ConfigurableJoint>();
            if (joint)
            {
                float x = 0, y = 0, z = 0;
                if (joint.angularXMotion != ConfigurableJointMotion.Locked)
                    x = continuousAct[++i];
                if (joint.angularYMotion != ConfigurableJointMotion.Locked)
                    y = continuousAct[++i];
                if (joint.angularZMotion != ConfigurableJointMotion.Locked)
                    z = continuousAct[++i];
                bodyPart.SetJointTargetFromRotVector(x, y, z);
            }
            else
            {
                Debug.LogWarning($"Wrong joint: {bodyPart.rb}");
            }
        }
    }

    private void FixedUpdate()
    {
        skeleton.RecordPrevState();

        AddReward(env.GetReward(jdController));
    }

    #endregion

}
