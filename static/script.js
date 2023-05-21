// Sayfa yüklendiğinde geri butonu öğesi bulunur ve tıklama işlemi dinlenir.
window.onload = function() {
  const backButton = document.getElementById('back-button');

  // Geri butonu öğesi tıklandığında geri gitme işlemi yapılır.
  backButton.addEventListener('click', function() {
    window.history.back();
  });
};
