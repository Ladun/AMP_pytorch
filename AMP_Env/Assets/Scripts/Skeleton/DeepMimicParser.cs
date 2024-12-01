using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace AMP
{
    public class DeepMimicParser
    {
        #region Class for parse json
        [Serializable]
        public class Humanoid3DData
        {
            public SkeletonData Skeleton;
            public BodyDefData[] BodyDefs;
            public DrawShapeDefData[] DrawShapeDefs;
        }

        [Serializable]
        public class SkeletonData
        {
            public JointData[] Joints;
        }

        [Serializable]
        public class JointData
        {
            public int ID;
            public string Name;
            public string Type;
            public int Parent;
            public float AttachX;
            public float AttachY;
            public float AttachZ;
            public float AttachThetaX;
            public float AttachThetaY;
            public float AttachThetaZ;
            public float LimLow0;
            public float LimHigh0;
            public float LimLow1;
            public float LimHigh1;
            public float LimLow2;
            public float LimHigh2;
            public float TorqueLim;
            public int IsEndEffector;
            public float DiffWeight;
        }

        [Serializable]
        public class BodyDefData
        {
            public int ID;
            public string Name;
            public string Shape;
            public float Mass;
            public int ColGroup;
            public int EnableFallContact;
            public float AttachX;
            public float AttachY;
            public float AttachZ;
            public float AttachThetaX;
            public float AttachThetaY;
            public float AttachThetaZ;
            public float Param0;
            public float Param1;
            public float Param2;
            public float ColorR;
            public float ColorG;
            public float ColorB;
            public float ColorA;
        }

        [Serializable]
        public class DrawShapeDefData
        {
            public int ID;
            public string Name;
            public string Shape;
            public int ParentJoint;
            public float AttachX;
            public float AttachY;
            public float AttachZ;
            public float AttachThetaX;
            public float AttachThetaY;
            public float AttachThetaZ;
            public float Param0;
            public float Param1;
            public float Param2;
            public float ColorR;
            public float ColorG;
            public float ColorB;
            public float ColorA;
        }
        #endregion

        public class Joint
        {
            public int id;
            public int parentId;
            public string name;
            public string type;
            public Vector3 attachPos;
            public Vector3 attachTheta;
            public Vector3 limLow;//radian
            public Vector3 limHigh; //radian
            public float torqueLim;

        }
        public class DrawShape
        {
            public int id;
            public string name;
            public string shape;
            public int parentId;
            public Vector3 param;
            public Vector3 attachPos;
            public Vector3 attachTheta;
            public Color color;
        }
        public class BodyShape
        {
            public int id;
            public string name;
            public string shape;
            public float mass;
            public Vector3 param;
            public Vector3 attachPos;
            public Vector3 attachTheta;
            public Color color;
        }

        public Dictionary<int, Joint> joints = new Dictionary<int, Joint>();
        public Dictionary<int, DrawShape> draws = new Dictionary<int, DrawShape>();
        public Dictionary<int, BodyShape> bodys = new Dictionary<int, BodyShape>();

       

        public void Parse(string skeletonFile, bool forced = true)
        {
            if(!forced)
            {
                if (joints.Count > 0)
                    return;
            }


            string path = Path.Combine(Utils.GetCurrentPath(), skeletonFile);
            string text = Utils.ReadTextFile(path);

            if (string.IsNullOrEmpty(text))
            {
                Debug.LogWarning($"Wrong text {skeletonFile}");
                return;
            }
            Humanoid3DData skeletonData = JsonUtility.FromJson<Humanoid3DData>(text);

            int sp = 0, re = 0, fi = 0;
            for (int i = 0; i < skeletonData.Skeleton.Joints.Length; i++)
            {
                Joint joint = new Joint();
                JointData jointData = skeletonData.Skeleton.Joints[i];
                joint.id = jointData.ID;
                joint.parentId = jointData.Parent;
                joint.name = jointData.Name;
                joint.type = jointData.Type;
                joint.attachPos = new Vector3(jointData.AttachX, jointData.AttachY, jointData.AttachZ);
                joint.attachTheta = new Vector3(jointData.AttachThetaX, jointData.AttachThetaY, jointData.AttachThetaZ);
                joint.limLow = new Vector3(jointData.LimLow0, jointData.LimLow1, jointData.LimLow2); 
                joint.limHigh = new Vector3(jointData.LimHigh0, jointData.LimHigh1, jointData.LimHigh2);
                joint.torqueLim = jointData.TorqueLim;

                if (joint.type == "revolute")
                    re += 1;
                else if (joint.type == "spherical")
                    sp += 1;
                else if (joint.type == "fixed" )
                    fi++;

                joints[joint.id] = joint;
            }
            Debug.Log($"Spherical: {sp}, Revolute: {re}, Fixed: {fi}");

            for (int i = 0; i < skeletonData.BodyDefs.Length; i++)
            {
                BodyShape body = new BodyShape();
                BodyDefData bodyDefData = skeletonData.BodyDefs[i];
                body.id = bodyDefData.ID;
                body.mass = bodyDefData.Mass;
                body.name = bodyDefData.Name;
                body.shape = bodyDefData.Shape;
                body.attachPos = new Vector3(bodyDefData.AttachX, bodyDefData.AttachY, bodyDefData.AttachZ);
                body.attachTheta = new Vector3(bodyDefData.AttachThetaX, bodyDefData.AttachThetaY, bodyDefData.AttachThetaZ);
                body.param = new Vector3(bodyDefData.Param0, bodyDefData.Param1, bodyDefData.Param2);
                body.color = new Color(bodyDefData.ColorR, bodyDefData.ColorG, bodyDefData.ColorB, bodyDefData.ColorA);

                bodys[body.id] = body;
            }

            for (int i = 0; i < skeletonData.DrawShapeDefs.Length;i++)
            {
                DrawShape draw = new DrawShape();
                DrawShapeDefData drawShapeDefData = skeletonData.DrawShapeDefs[i];
                draw.id = drawShapeDefData.ID;
                draw.name = drawShapeDefData.Name;
                draw.shape = drawShapeDefData.Shape;
                draw.parentId = drawShapeDefData.ParentJoint;
                draw.attachPos = new Vector3(drawShapeDefData.AttachX, drawShapeDefData.AttachY, drawShapeDefData.AttachZ);
                draw.attachTheta = new Vector3(drawShapeDefData.AttachThetaX, drawShapeDefData.AttachThetaY, drawShapeDefData.AttachThetaZ);
                draw.param = new Vector3(drawShapeDefData.Param0, drawShapeDefData.Param1, drawShapeDefData.Param2);
                draw.color = new Color(drawShapeDefData.ColorR, drawShapeDefData.ColorG, drawShapeDefData.ColorB, drawShapeDefData.ColorA);

                draws[draw.id] = draw;
            }
        }

    }
}