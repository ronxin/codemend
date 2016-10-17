$(document).ready(function () {
  // Get code from brain (for user testing)
  $.getJSON('brain-request', {'type': 'experiment_mode'},
      function(response) {
        experiment_mode = response.mode;  // global
        init_codemend();
      }
    );

  resize_window();

  // global variables
  editorSelectionBackup = null;
});

$(window).resize(resize_window);

var query_send_timeout = null;  // wait a certain amount of time before sending query

function resize_window() {
  var w = $(window);
  var w_effective_width = w.width() - 50;
  var left_width = Math.min(w_effective_width * 0.5 - 15, 650);
  var right_width = w_effective_width - left_width - 15;
  var w_effective_height = w.height() * 0.95;
  var left_usable_height = w_effective_height - $("#app-header").outerHeight();
  var svg_div_height = Math.min(left_usable_height * 0.5, 400);
  var svg_height = svg_div_height;
  var editor_div_height = left_usable_height - svg_div_height;
  var editor_height = editor_div_height - $(".code-helper-banner").outerHeight() - $("#btn-execute").outerHeight();
  var right_main_content_height = w_effective_height - $("#top-right-query-box").outerHeight();

  $("#left-column").width(left_width);
  $("#right-column").width(right_width);
  $("#top-left-panel").height(editor_div_height);
  $("#bottom-left-panel").height(svg_div_height);
  $("#top-left-panel .CodeMirror").height(editor_height);
  $("#bottom-left-panel svg").attr('height', svg_height);
  $("#right-main-content").height(right_main_content_height);
}

function init_codemend() {
  if (experiment_mode != 'no_mode') {
    $('.hide-in-user-study').hide();
  }

  editor = CodeMirror.fromTextArea(
    document.getElementById('code-text'), {
      mode: "python",
      theme: "default",
      lineNumbers: true,
      readOnly: false,
      viewportMargin: Infinity,
      highlightSelectionMatches: {showToken: /\w/, annotateScrollbar: true},
    });

  editor.setValue(' ');  // to "activate" the editor height setting.

  editor.setOption('extraKeys', {
    'Ctrl-Enter': sendPltRequest
  });

  editor.on('cursorActivity', onCursorActivity);
  editor.on('focus', function() {
    editor.ignore_cursor_event = false;
    onCursorActivity();
  });

  $('#btn-execute').click(sendPltRequest);

  $('.nl-text-field').keyup(function (e) {
    clearTimeout(query_send_timeout);

    if (e.keyCode == 13) {
      // NL query: Press Enter -> Submit + blur
      sendNlpRequest();
      sendGoogleRequest();
      this.blur();
    }
    else if (e.keyCode == 27) {
      // NL query: Press Escape -> blur
      $('.nl-text-field').val('');
      this.blur();
    }
    else {
      query_send_timeout = setTimeout(function() {
        sendNlpRequest();
        sendGoogleRequest();
      }, 1000);
    }

    if (e.keyCode == 32) {
      // NL query: Press Space -> Submit
      sendNlpRequest();
    }
  }).on("click", function() {
    // NL query: On click -> Select All
    $(this).select();
    clearTimeout(query_send_timeout);
  });

  // Key (printable chars) pressed outside any text area -> focus on the NL query.
  $(document).keypress(function(e) {
    if ($("input,textarea").is(":focus")) return;
    var keyCode = e.keyCode;
    if (keyCode >= 33 && keyCode <= 126) {
      var charCode = e.charCode;
      var charc = String.fromCharCode(charCode);
      $('.nl-text-field').val('').focus();
    }
  });

  // Escape pressed outside any text area -> clear the NL query
  $(document).keyup(function(e) {
    if (e.keyCode == 27) {
      $('.nl-text-field').val('');
    }
  })

  // This will trigger the rendering of the default code sample.
  populateCodeSampleList();

  // list of current markers - global
  codeMirrorMarkers = [];

  // list of current line widgets - global
  codeMirrorLineWidgets = [];
}

function setCurrentCode(code) {
  // TODO: use replaceRange when possible (requires backend modification)
  cursorPosition = editor.getCursor();
  editor.setValue(code + '');
  editor.setCursor(cursorPosition);
}

function getCurrentCode() {
  return editor.getValue();
}

function getCurrentNLQuery() {
  return $('.nl-text-field').val();
}

function setCurrentNLQuery(query) {
  $('.nl-text-field').val(query);
}

function populateCodeSampleList() {
  if (experiment_mode == 'no_mode') {
    var presets = ['user-study/practice',
                   'user-study/task1',
                   'user-study/task2',
                   'helloworld', 'empty','casestudy1',
                   'casestudy2','formattest','shadow',
                   'eval1','eval2','eval3','eval4','eval5'];

    var select = d3.select("#sel-presets");

    var options = select.selectAll("option")
      .data(presets)
      .enter()
      .append("option")
      .attr("value", function(d) {return d})
      .html(function(d) {return d});

    select.on("change", function() {
      var selectedIndex = select.property("selectedIndex");
      var selectedPresetName = presets[selectedIndex];
      // var selectedPreset = options.filter(function(d,i) {return i == selectedIndex});
      // var selectedPresetName = selectedPreset.datum().data;
      updateCodeSample(selectedPresetName);
    });

    updateCodeSample(presets[0]);  // the default plot
  } else {
    var modeId = experiment_mode[1];
    if (modeId == 0) {
      updateCodeSample('user-study/practice');
    } else if (modeId == 1) {
      updateCodeSample('user-study/task1');
    } else if (modeId == 2) {
      updateCodeSample('user-study/task2');
    }
  }
}

function updateCodeSample(presetName) {
  var fileUrl = 'code-samples/' + presetName + '.py';
  $.get(fileUrl, {}, function(data) {
    setCurrentCode(data);
    sendPltRequest();
  });
}

function sendPltRequest() {
  var code = getCurrentCode();
  $('#bottom-error-plt').empty().hide();
  $.getJSON('plt-request', {'code':code}, processPltResult);
  $("#loader-gif-plt").show();
}

function processPltResult(data) {
  $("#loader-gif-plt").hide();
  if (data.hasOwnProperty('error')) {
    $('#bottom-error-plt').append(data.error).show();
  } else {
    var svg = $(data.svg);
    current_svg = svg;  // global: used for certain mouseout behaviors
    updateSVG(svg);
  }
}

function sendNlpRequest() {
  if (experiment_mode[0] == 'g') return;

  var query = getCurrentNLQuery();
  if (query.trim().length == 0) return;

  var code = getCurrentCode();
  $('#bottom-error-nlp').empty().hide();
  $.getJSON('brain-request',
    {'code': code, 'query': query, 'type': 'nlp'},
    handleBrainResponse);
  $("#loader-gif-nlp").show();

  sendSummaryRequest();
}

function updateExampleGallery(examples) {
  $("#example-gallery").empty();
  d3.select("#example-gallery")
    .selectAll("span")
    .data(examples)
    .enter()
    .append('span')
    .attr('class', 'example-gallery-item')
    .html(function(d) {
      return d.svg;
    })
    .select('svg')
    .attr('width', '100%')
    .attr('height', '200px')
    .on('click', function(d) {
      setCurrentCode(d.code);
      sendPltRequest();
      // window.scrollTo(0,0);
      updateExampleGallery([]);
    });

  d3.selectAll("#example-gallery span")
    .style('width', function(d) {
      // The computed size of SVG
      var width = d3.select(this).select('g').node().getBoundingClientRect().width + 10;
      // Don't be too wide
      var width = Math.min(width, $(window).width() / 5);
      return width + 'px';
    });

  resize_window();
}

function updateSVG(svg) {
  /*
  Updates the top-right SVG window.

  Got help from this very helpful tutorial: https://css-tricks.com/scale-svg/
  */
  var svg = $(svg);
  svg.removeAttr('width')
    .removeAttr('height')
    .attr('width', "100%");

  // svg.attr("preserveAspectRatio", "xMinYMin meet");

  $('#plt-render-box').empty().append(svg);
  resize_window();  // to properly adjust svg height.
}

function clearEditorHighlights() {
  // Clear line backgrounds
  for (var i = 0; i < editor.getDoc().size; i++) {
    editor.removeLineClass(i, "background");
  }

  // Clear marker (code tokens) backgrounds
  for (var i = 0; i < codeMirrorMarkers.length; i++) {
    codeMirrorMarkers[i].clear();
  }
  codeMirrorMarkers = [];

  // Clear line widgets (for code suggestion)
  for (var i = 0; i < codeMirrorLineWidgets.length; i++) {
    codeMirrorLineWidgets[i].clear();
  }
  codeMirrorLineWidgets = [];
}

function weight2ColorClass(weight) {
  var intClassNumber = Math.round(weight);
  if (intClassNumber > 0 && intClassNumber <= 10) {
    return 'code-mine-bg-' + intClassNumber;
  }
  return false;
}

function onCursorActivity() {
  if (editor.ignore_cursor_event) return;
  if (typeof cursorTimeout != "undefined") clearTimeout(cursorTimeout);

  cursorTimeout = setTimeout(function() {
    sendSummaryRequest();
  }, 200);

  // comment-triggered search
  var cursor = editor.getCursor()
  var curLine = editor.getLine(cursor.line).trim();
  if (curLine.length >= 2 && curLine[0] == '#') {
    var query = curLine.replace(/^[\s#]+/g, '');
    setCurrentNLQuery(query);
    if (curLine[cursor.ch - 1] == ' ') {
      sendNlpRequest();
    }
    clearTimeout(query_send_timeout);
    query_send_timeout = setTimeout(function() {
        sendNlpRequest();
        sendGoogleRequest();
      }, 500);
  }
}

function sendSummaryRequest() {
  var request = prepareBasicRequest('summary')
  $.getJSON('brain-request', request, handleBrainResponse);
}

function handleBrainResponse(data) {
  $('#bottom-error-brain').empty();

  if (data.type == 'nlp') {
    $("#loader-gif-nlp").hide();
  }

  if (data.error) {
    handleBrainError(data)
    return;
  }

  // The following handling mostly does not distinguish the type of request.
  // It deals with all kinds of content accordingly, thus allowing each
  // response type to contain mixed content types.

  if (data.matches) {  // NLP Request
    var matches = data.matches;
    clearEditorHighlights();
    for (var i = 0; i < matches.length; i++) {
      var match = matches[i];

      // For function only: line marking
      if (match.type == 'func') {
        var lineWeightScaled = match.weight * 2;
        var lineColorClass =  weight2ColorClass(lineWeightScaled);
        if (lineColorClass) {
          editor.addLineClass(match.lineno, "background", lineColorClass)
        }
      }

      // For everything else: chunk marking
      else {
        var weightScaled = match.weight * 5;
        var colorClass = weight2ColorClass(weightScaled);
        if (colorClass) {
          var marker = editor.markText(
            {line: match.lineno, ch: match.col_offset},
            {line: match.end_lineno, ch: match.end_col_offset},
            {className:  colorClass});
          codeMirrorMarkers.push(marker);
        }
      }
    }
  }

  if (data.summary_groups) {  // Summary Request
    d3.select('#summary-box').selectAll('.summary-group').remove();
    renderSummaryGroups(d3.select("#summary-box"), data.summary_groups, 0);
  }
}

function prepareBasicRequest(request_type) {
  var cur = editor.getCursor();
  var code = getCurrentCode();
  var query = getCurrentNLQuery();
  var request = {'code': code, 'cursor_line': cur.line,
                 'cursor_ch': cur.ch, 'type': request_type,
                 'query': query};
  return request;
}

function handleBrainError(data) {
  if (data.error == 'syntax error') return;
  $('#bottom-error-brain').empty();
  $('#bottom-error-brain').append(data.error).show();
}

/*
  Renders a list of summary groups as boxes. Supports rendering boxes inside
  another box.

  Parameters:
  - rootSelection: d3 selection in which the summary groups are to be rendered
  - data: straight from backend, corresponding to a list of summary groups.
  - level
*/
function renderSummaryGroups(rootSelection, data, level) {
  for (var i = 0; i < data.length; i++) {
    var selection = rootSelection.insert('div', '.google-result')
      .classed("summary-group", true)
      .classed('level-'+level, true);
    renderSummaryGroup(selection, data[i], level);
  }

  resize_window();  // make sure new items have proper sizes
}

function appendDocStringDiv(selection, d) {
  // DocString
  var docstring = d.docstring || d.doc;
  if (docstring) docstring = '<b>' + docstring + '</b>';
  if (d.signature && d.signature != '()') docstring = '' + d.signature + '<br>' + docstring;
  if (!docstring) docstring = 'No docstring available';
  selection.append('div')
    .html(docstring)
    .classed('docstring', true)
    .classed('more', true);
  updateMoreLess();  // "see more" & "see less" links in moreless.js
}

/*
  Renders a single summary group.
*/
function renderSummaryGroup(selection, d, level) {
  selection.selectAll('*').remove();

  if (d.name != 'TOP_MODULE') {
    // Group Name
    selection.append('h3')
      .text(d.name);

    appendDocStringDiv(selection, d);
  }

  if (d.suggest) {
    if (d.val_text_range) d.suggest.val_text_range = d.val_text_range;
    renderSuggestGroup(selection, d.suggest, level, null);
  }
}

/*
  Renders a single suggest group.

  Parameters:
  - tileSelection: the d3 selection of the tile that *spawns* this suggest
    group. Could be null if the suggest group is not spawned by a tile.
*/
function renderSuggestGroup(rootSelection, data, level, tileSelection) {
  var suggestType = data.type;
  rootSelection.selectAll('.suggest-group').remove();
  var selection = rootSelection.append('div')
    .classed('suggest-group', true)
    // always the same level as the parent summary-group
    .classed('level-' + level, true)
    .datum(data);

  selection.datum().tileSelection = tileSelection;

  // Function call preview
  if (suggestType == 'parameters') {
    var cmContainerMaster = selection.append('div')
      .classed('cm-container', true);

    // preview title
    if (data.mother_elem_type == 'func') preview_title_text = 'Function call preview';
    else preview_title_text = 'Argument configuration preview';

    cmContainerMaster.append('div').text(preview_title_text);

    // preview code editor
    var cmContainer = cmContainerMaster.append('div')
      .classed('tmp', true);
    var tmpCodeMirror = CodeMirror(cmContainer[0][0], {
      value: "",
      mode: 'python',
      viewportMargin: Infinity,
      lineWrapping: true,
      readOnly: false// 'nocursor',
    });
    selection.datum().call_preview = {
      editor: tmpCodeMirror,
      root_id: data.elem_id,
      root_name: data.elem_name,
      params: [],
      mother_elem_type: data.mother_elem_type
    };
    if (!data.is_to_insert) {
      for (var i=0; i<data.items.length; i++) {
        var item = data.items[i];
        if (item.used_val) {
          selection.datum().call_preview.params.push(
              {key: item.val, value: item.used_val}  // generated in code_suggest.get_arg_suggests
          );
        }
      }
    }
    updateCallPreview(null, selection, [], 'add');

    // preivew accept button
    cmContainerMaster.append('a')
      .attr('href', '#')
      .attr('class', 'btn btn-success')
      .text(function() {
        return data.is_to_insert ? 'Insert' : 'Update';
      })
      .on('click', function() {
        var tmpEditorValue = tmpCodeMirror.getValue();
        if (data.is_to_insert) {
          // TODO: replace this with more principled way of line insertion --
          // allowing the user to select where to insert.
          var lines = getCurrentCode().split('\n');
          var bestLineNo = data.best_position - 1;  // convert to 0-index
          lines.splice(bestLineNo, 0, tmpEditorValue);
          setCurrentCode(lines.join('\n'));
          sendPltRequest();
        } else {
          var r = data.mother_text_range;
          if (r) {
            editor.ignore_cursor_event = true;
            editor.setSelection({line: r[0]-1, ch: r[1]}, {line: r[2]-1, ch: r[3]});
            editor.replaceSelection(tmpEditorValue, 'around');

            r[0] = editor.getCursor('anchor').line + 1;
            r[1] = editor.getCursor('anchor').ch;
            r[2] = editor.getCursor('head').line + 1;
            r[3] = editor.getCursor('head').ch;

            sendPltRequest();
          }
        }
      });
  }

  // Suggest Group Title
  if (data.items && data.items.length > 0) {
    selection.append('div')
      .classed('suggest-group-title', true)
      .classed('level-' + level, true)
      .text(data.name);

    // Suggest tiles (pills)
    var tiles = selection.append('div')
      .classed('suggest-group-body', true)
      .selectAll("span")
      .data(data.items)
      .enter()
      .append("span")
      .text(function(d) {return d.val})
      .attr("class", function(d) {
        var weight = d.weight;  // [0,1]
        var weightScaled = weight * 10;  // [0, 10]
        return weight2ColorClass(weightScaled);
      })
      .classed('code-suggest-tile', true)
      .classed('level-'+level, true)
      .classed('checked', function(d) {return d.used_val})
      .on('mouseover', function(d) {
        handleSuggestTileMouseEvent('mouseover', d, this, selection, level, d.mode);
      })
      .on('mouseout', function(d) {
        handleSuggestTileMouseEvent('mouseout', d, this, selection, level, d.mode);
      })
      .on('click', function(d) {
        handleSuggestTileMouseEvent('click', d, this, selection, level, d.mode);
      });

    // Delete buttons in suggest tiles
    tiles.append('span')
      .html('&nbsp;&#10006;')
      .attr('title', 'Delete')
      .classed('delete-button', true)
      .on('click', function(d) {
        updateCallPreview(d, selection, [], 'delete');
        d3.select(this.parentNode).classed('checked', false);
        d3.event.stopPropagation();
      });
  }
}

function handleSuggestTileMouseEvent(event, d, tileNode, suggestGroupSelection, level, mode) {
  var tileSelection = d3.select(tileNode);
  if (mode == 'gallery') {
    if (event == 'mouseover') {
      if (d3.select('.code-suggest-tile.fixed').empty()) {
        updateExampleGallery(d.examples || []);
      }
    }

    else if (event == 'mouseout') {
      if (d3.select('.code-suggest-tile.fixed').empty()) {
        updateExampleGallery([]);
      }
    }

    else if (event == 'click') {
      if (tileSelection.classed('fixed')) {
        tileSelection.classed('fixed', false);
      } else {
        tileSelection.classed('fixed', true)
        updateExampleGallery(d.examples || []);
      }
    }
  }

  else if (mode == 'non_terminal') {
    if (event == 'click') {
      expandTile(d, tileSelection, level, mode)
    }
  }

  else if (mode == 'terminal') {
    if (level == 0) {
      var r = suggestGroupSelection.datum().val_text_range;
      if (event == 'mouseover') {
        if (r) {
          editor.ignore_cursor_event = true;
          editor.setSelection({line: r[0]-1, ch: r[1]}, {line: r[2]-1, ch: r[3]});
          editorSelectionBackup = editor.getSelection();
          editor.replaceSelection(d.val, 'around');
        }

      }

      else if (event == 'mouseout') {
        if (editorSelectionBackup) {
          editor.replaceSelection(editorSelectionBackup);
          editorSelectionBackup = null;
        }
      }

      else if (event == 'click') {
        if (r) {
          r[0] = editor.getCursor('anchor').line + 1;
          r[1] = editor.getCursor('anchor').ch;
          r[2] = editor.getCursor('head').line + 1;
          r[3] = editor.getCursor('head').ch;
        }
        editor.ignore_cursor_event = true;
        if (editorSelectionBackup) {
          editorSelectionBackup = null;
          sendPltRequest();
        }
      }
    }

    else {
      if (event == 'click') {
        updateCallPreview(d, suggestGroupSelection, [], 'add')
      }
    }
  }

  else {
    console.log("Error: unrecognized mode = " + mode);
    console.log(d);
  }
}

function expandTile(d, tileSelection, level, mode) {
  var tileNode = tileSelection.node();
  var parentSelection = d3.select(tileSelection.node().parentNode);
  d3.selectAll('.code-suggest-detail-info.level-'+level).remove();
  if (tileSelection.classed('active')) {
    tileSelection.classed('active', false);
  } else {
    d3.selectAll('.code-suggest-tile.level-'+level+'.active').classed('active', false);
    d3.selectAll('.code-suggest-tile.level-'+level+'.lower-first').classed('lower-first', false);
    tileSelection.classed('active', true);
    var current_top = tileNode.getBoundingClientRect().top;
    var lowers = d3.selectAll('.code-suggest-tile.level-'+level).filter(function(d) {
      return d3.select(this).node().getBoundingClientRect().top > current_top;
    });
    if (lowers) {
      d3.select(lowers[0][0]).classed('lower-first', true);
    }
    var detailed_box = parentSelection
      .insert('div', '.code-suggest-tile.level-'+level+'.lower-first')
      .classed('code-suggest-detail-info', true)
      .classed('level-'+level, true);

    appendDocStringDiv(detailed_box, d);

    if (mode == 'non_terminal') {
      var request = prepareBasicRequest('suggest');
      request.elem_id = d.elem_id;
      $.getJSON(
        'brain-request', request, function(response) {
          if (response.error) {
            handleBrainError(response);
            return;
          }
          renderSuggestGroup(detailed_box, response.suggest, level + 1, tileSelection);
        }
      );
    }
  }
}

/*
  Pass on tileData to ancestors, until reaching a suggest group that holds the
  actual code preview, at which point the code preview will be updated.

  Parameters:
  - tileData: the data to pass
  - suggestGroupSelection: the suggest group that contains this tile
  - dataBelow: data below this tile
  - action: 'add' | 'delete'
*/
function updateCallPreview(tileData, suggestGroupSelection, dataBelow, action) {
  if (tileData) {
    dataBelow.push(tileData);
  }
  // memory state of the preview box (defined in renderSuggestGroup)
  var call_preview = suggestGroupSelection.datum().call_preview;
  if (call_preview) {  // This suggest group holds the preview code editor
    if (dataBelow) { // Update params using dataBelow
      paramLookup = {};  // [param] = idx
      for (var i = 0; i < call_preview.params.length; i++) {
        paramLookup[call_preview.params[i].key] = i
      }
      // TODO: handle situation where dictionary is value
      if (dataBelow.length == 2) {  // add/change argval
        var newParam = {key: dataBelow[1].val, value: dataBelow[0].val};
        if (newParam.key in paramLookup) {
          var idx = paramLookup[newParam.key];
          call_preview.params[idx].value = newParam.value;
        } else {
          call_preview.params.push(newParam);
        }
      } else if (dataBelow.length == 1) {  // remove arg
        var key = dataBelow[0].val;
        if (key in paramLookup) {
          var idx = paramLookup[key];
          call_preview.params[idx] = null;  // mark as to-be-deleted
        }
      }
      // delete marked params
      call_preview.params = call_preview.params.filter(function(d) {return d != null});
    }
    if (call_preview.mother_elem_type == 'func') {
      if (call_preview.root_id) {
        var code = call_preview.root_id + '(';
        call_preview.params.sort(function(a, b) {
          if (isNumeric(a.key) && !isNumeric(b.key)) return -1;
          else if (!isNumeric(a.key) && isNumeric(b.key)) return 1;
          else if (isNumeric(a.key) && isNumeric(b.key)) return parseInt(a.key) - parseInt(b.key);
          else return a.key < b.key ? -1 : 1;
        });
        for (var i = 0; i < call_preview.params.length; i++) {
          var key = call_preview.params[i].key;
          var value = call_preview.params[i].value;
          if (i > 0) code += ', ';
          if (!isNumeric(key)) {
            code += key;
            code += '=';
          }
          code += value;
        }

        code += ')';
        call_preview.editor.setValue(code);
      }
    } else if (call_preview.mother_elem_type == 'arg') {
      var code = '';
      if (call_preview.root_name && !isNumeric(call_preview.root_name)) {
        code += call_preview.root_name + '=';
      }
      code += '{';
      for (var i = 0; i < call_preview.params.length; i++) {
        if (i > 0) code += ', ';
        var key = call_preview.params[i].key;
        var value = call_preview.params[i].value;
        code += "'" + key + "': " + value;
      }
      code += '}';
      call_preview.editor.setValue(code);
    }
  } else {
    var parent_sg = $(suggestGroupSelection.node().parentNode).closest('.suggest-group');
    if (parent_sg) {
      var parent_sg_selection = d3.select(parent_sg[0]);
      var myTileSelection = suggestGroupSelection.datum().tileSelection;
      var myTileData = myTileSelection.datum();
      if (action == 'add') { //&& parent_sg_selection.datum().mode == 'non_terminal') {
        myTileSelection.classed('checked', true);
      }
      updateCallPreview(myTileData, parent_sg_selection, dataBelow, action);
    }
  }
}

function sendGoogleRequest() {
  var query = getCurrentNLQuery();
  $.getJSON('brain-request',
    {'query': query + ' matplotlib', 'type': 'google'},
    handleGoogleResponse);
}

function handleGoogleResponse(response) {
  return;  // google response now disabled

  console.log(response);
  var summary_box = d3.select('#summary-box');
  summary_box.selectAll('.google-result').remove();
  var google_result_box = summary_box.append('div')
    .classed('google-result', true);

  google_result_box.append('div')
    .classed('google-result-title', true)
    .text('Google: "' + response.query + '"');

  var results = response.results || [];
  var children = google_result_box.append('div')
    .classed('google-result-body', true)
    .selectAll('div')
    .data(results)
    .enter()
    .append('div')
    .classed('google-result-child', true);

  children.append('div')
    .html(function(d) {return '<a href="' + d.url + '" target="_blank">' + d.title + '</a>'});

  children.append('div')
    .html(function(d) {return '<a href="' + d.url + '" target="_blank">' + d.visibleUrl + '</a>'});

  children.append('div')
    .html(function(d) {return d.content});
}

// http://stackoverflow.com/questions/1418050/string-strip-for-javascript
if (typeof(String.prototype.trim) === "undefined")
{
    String.prototype.trim = function()
    {
        return String(this).replace(/^\s+|\s+$/g, '');
    };
}

function isNumeric(n) {
  return !isNaN(parseFloat(n)) && isFinite(n);
}
