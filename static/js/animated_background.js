const feelingColorMap = {
    1: ["#00204a", "#9896f1", "#00bbf0"],
    2: ["#edf756", "#ffa8B6", "#0049B7"], 
    3: ["#e5eaf5", "#12343b", "#7d3cff"] 
  };
  
  function setBackgroundColors(feeling) {
    const [a, b, c] = feelingColorMap[feeling];
    document.documentElement.style.setProperty("--color-a", a);
    document.documentElement.style.setProperty("--color-b", b);
    document.documentElement.style.setProperty("--color-c", c);
  }
  
  let currentFeeling = 1;
  setInterval(() => {
    currentFeeling = (currentFeeling % 3) + 1;
    setBackgroundColors(currentFeeling);
  }, 5000);

  document.addEventListener('DOMContentLoaded', function() {
    var feeling = parseInt(document.documentElement.getAttribute('data-feeling')) || 1;
    setBackgroundColors(feeling);
});