import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test l'endpoint de santé"""
    print("🔄 Test /health...", end=" ")
    try:
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        print("✅ OK")
        print(f"   → Statut: {data['status']}")
        print(f"   → Modèle chargé: {data['model_loaded']}")
        return True
    except Exception as e:
        print(f"❌ ÉCHEC: {e}")
        return False

def test_predict():
    """Test l'endpoint de prédiction"""
    print("\n🔄 Test /predict...", end=" ")
    try:
        payload = {
            "resume_text": "Développeur Python avec 5 ans d'expérience en machine learning",
            "job_description": "ML Engineer senior pour projet de classification"
        }
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "decision" in data
        
        print("✅ OK")
        print(f"   → Décision: {data['decision']}")
        if "confidence" in data:
            print(f"   → Confiance: {data['confidence']}")
        return True
    except Exception as e:
        print(f"❌ ÉCHEC: {e}")
        return False

def test_invalid_input():
    """Test avec des entrées invalides"""
    print("\n🔄 Test entrée invalide...", end=" ")
    try:
        payload = {
            "resume_text": ""
        }
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        # Doit retourner une erreur 422 (validation échouée)
        assert response.status_code == 422
        print("✅ OK (validation fonctionne)")
        return True
    except Exception as e:
        print(f"❌ ÉCHEC: {e}")
        return False

def test_root():
    """Test l'endpoint racine"""
    print("\n🔄 Test /...", end=" ")
    try:
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ ÉCHEC: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 LANCEMENT DES TESTS DE L'API")
    print("=" * 50)
    print("⚠️  Assurez-vous que l'API tourne (python app.py)")
    print("=" * 50)
    
    time.sleep(1)
    
    results = []
    results.append(test_root())
    results.append(test_health())
    results.append(test_predict())
    results.append(test_invalid_input())
    
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    print(f"✅ Passés: {passed}/{total}")
    print(f"❌ Échoués: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 TOUS LES TESTS ONT RÉUSSI !")
    else:
        print("\n⚠️  Certains tests ont échoué, vérifiez l'API.")