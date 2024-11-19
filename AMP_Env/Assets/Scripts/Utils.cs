using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace AMP
{
    public class Utils
    {

        public static string ReadTextFile(Object file)
        {
            if(file == null)
            {
                Debug.LogWarning("File not assigned!");
                return "";
            }

            string filePath = AssetDatabase.GetAssetPath(file);
            if (File.Exists(filePath))
            {
                string fileContent = File.ReadAllText(filePath);
                return fileContent;
            }
            else
            {
                Debug.LogError("File does not exist at path: " + filePath);
            }

            return "";
        }
    }
}