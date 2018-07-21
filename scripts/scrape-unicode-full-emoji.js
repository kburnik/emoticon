var $headings = $(".mediumhead").map(function() {
  return $(this).parent().get();
});

/*
$headings.each(function(){
  var $h = $(this);
  var groupName = $h.find("a").html();
  console.log(groupName);
})
*/

String.prototype.replaceAll = function(search, replacement) {
    var target = this;
    return target.split(search).join(replacement);
};

function createCanvas() {
  var canvas = document.createElement('canvas');
  $("#canvas").remove();
  canvas.id = "canvas";
  canvas.width = 72;
  canvas.height = 72;

  canvas.style.zIndex = 1000;
  canvas.style.position = "fixed";
  canvas.style.border = "20px solid";
  canvas.style.left = 0;
  canvas.style.top = 0;
  canvas.style.width = canvas.width;
  canvas.style.height = canvas.height;
  $("body").prepend($(canvas));
  return canvas;
}

function download(filename, canvas) {
  var link = document.createElement('a');
  link.download = filename;
  link.href = canvas.toDataURL('image/png');
  document.body.appendChild(link);
  link.click();
  link.parentNode.removeChild(link);
}

var canvas = createCanvas();

for (var i = 0; i < $headings.length - 1; i++) {
  var $h = $($headings[i]);
  var groupName = $h.find("a").html();
  var $rows = $h.nextUntil($headings[i + 1]).slice(1);

  if (!groupName.startsWith('face-')) continue;

  groupName = groupName.replaceAll(' ', '-');


  $rows.each(function() {
    var $r = $(this);
    var className = $r.find("td.name").html();
    if (!className) return;
    className = className.replaceAll(' ', '-');

    var $images = $r.find("td.andr").slice(0, 7).find("img");
    $images.each(function() {
      var $img = $(this);
      var index = $images.get().indexOf($img[0]);
      var filename = [
          groupName,
          className,
          className + '-' + index + ".png"].join("__");
      console.log(filename);

      var ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.beginPath();
      ctx.rect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "white";
      ctx.fill();

      ctx.drawImage($img[0], 0, 0);
      // download(filename, canvas);

      // debugger;
    })

    // console.log(className, $images.length, $images);
  })
}
