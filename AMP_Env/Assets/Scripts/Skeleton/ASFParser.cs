using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace AMP
{

    public class ASFParser
    {
        public class Bone
        {
            public int id;
            public string name;
            public Vector3 direction;
            public float length;
            public Vector3 axis;
            public List<string> dof = new List<string>();
            public Vector2[] limits;
            public string parentName;
        }

        public Dictionary<string, Bone> bones = new Dictionary<string, Bone>();
        public float lengthScale = 1;

        public void ParseASF(UnityEngine.Object skeletonFile)
        {
            string text = Utils.ReadTextFile(skeletonFile);

            if (string.IsNullOrEmpty(text))
            {
                Debug.LogWarning($"Wrong text {skeletonFile}");
                return;
            }
            string[] lines = text.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None)
                                .Where(s => !string.IsNullOrWhiteSpace(s))
                                .ToArray();

            Bone currentBone = null;
            //0 == None, 1 == unit, 2 == bonedata, 3 == hierarchy
            int currentDataType = 0;

            foreach (string line in lines)
            {
                string trimmedLine = line.Trim();
                if (trimmedLine.StartsWith(":unit"))
                {
                    currentDataType = 1;
                    continue;
                }
                else if (trimmedLine.StartsWith(":bonedata"))
                {
                    currentDataType = 2;
                    continue;
                }
                else if (trimmedLine.StartsWith(":hierarchy"))
                {
                    currentDataType = 3;
                    continue;
                }

                switch (currentDataType)
                {
                    case 1:
                        if (trimmedLine.StartsWith("length"))
                        {
                            string[] parts = trimmedLine.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                            if (parts.Length >= 2 && float.TryParse(parts[1], out float scale))
                            {
                                lengthScale = scale;
                                break;
                            }
                        }

                        break;
                    case 2:
                        if (trimmedLine.StartsWith("begin"))
                        {
                            currentBone = new Bone();
                        }
                        else if (trimmedLine.StartsWith("end"))
                        {
                            bones[currentBone.name] = currentBone;
                            currentBone = null;
                        }
                        else if (currentBone != null)
                        {
                            string[] parts = trimmedLine.Split(new[] { ' ' }, 2);
                            switch (parts[0])
                            {
                                case "id":
                                    currentBone.id = int.Parse(parts[1]);
                                    break;
                                case "name":
                                    currentBone.name = parts[1];
                                    break;
                                case "direction":
                                    {
                                        string[] p = parts[1].Split(' ');
                                        currentBone.direction = new Vector3(
                                            float.Parse(p[0]),
                                            float.Parse(p[1]),
                                            float.Parse(p[2])
                                        ).normalized;
                                        break;
                                    }
                                case "length":
                                    currentBone.length = float.Parse(parts[1]);
                                    break;
                                case "axis":
                                    {
                                        string[] p = parts[1].Split(' ');
                                        currentBone.axis = new Vector3(
                                            float.Parse(p[0]),
                                            float.Parse(p[1]),
                                            float.Parse(p[2])
                                        );
                                        break;
                                    }
                                case "dof":
                                    currentBone.dof = parts[1].Split(' ').ToList();
                                    break;
                                case "limits":
                                    currentBone.limits = ParseLimits(parts[1]);
                                    break;
                            }
                        }
                        break;
                    case 3:
                        if (!trimmedLine.StartsWith("begin") && !trimmedLine.StartsWith("end"))
                        {
                            string[] parts = trimmedLine.Split(' ');

                            for (int i = 1; i < parts.Length; i++)
                            {
                                bones[parts[i]].parentName = parts[0];
                            }
                        }
                        break;
                }
            }
        }

        private Vector2[] ParseLimits(string s)
        {
            List<Vector2> limitsList = new List<Vector2>();
            string[] parts = s.Split(new[] { '(', ')' }, StringSplitOptions.RemoveEmptyEntries);

            foreach (string part in parts)
            {
                string[] limits = part.Trim().Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (limits.Length >= 2)
                {
                    float low, high;
                    if (float.TryParse(limits[0], out low) && float.TryParse(limits[1], out high))
                    {
                        limitsList.Add(new Vector2(low, high));
                    }
                }
            }

            return limitsList.ToArray();
        }
    }

}