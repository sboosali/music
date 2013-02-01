#!/usr/bin/python

"""
A simple python MIDI file player using PyGame and PyGTK/Hildon

Intended for Nokia Internet Tablet

Version 0.2. Copyright (c) 1 Jan 2008. Gek S. Low. All right reserved.


Changes:
- 0.1 (25 Dec 07) - Initial release
- 0.2 (1 Jan 08) - Added custom file browser (faster performance than GTK FileChooser), and basic playlist functionality

"""

import pygtk
pygtk.require('2.0')
import gtk
import gobject

import pygame
import sys
import os.path

try:
	import hildon
	isHildon = True
	print "Is Hildon!"
except:
	# No hildon module, default is gtk
	print "Not Hildon!"
	isHildon = False


APP_TITLE = "PlayMusic 0.2"
APP_ABOUT = "Python MIDI player using PyGame\n\n\
Version 0.2. Copyright (c) 2008 Gek S. Low. All rights reserved.\n\n\
Double-clicking on a filename will play the file immediately. You can also add the file to the playlist. Stopping the current song will automatically skip to the next song in the playlist."

class PlayMus:
	def displayAlert(self, text):
		alert = gtk.MessageDialog()
		alert.set_markup(text)

	def displayStatus(self, text):
		self.statusLabel.set_text("\n" + text + "\n")

	def doIdle(self):
		if pygame.mixer.music.get_busy():
			# Display time elapsed if playing a song
			# convert to minutes and seconds
			mstime = pygame.mixer.music.get_pos()
			minutes = int(mstime / 60000)
			seconds = int(mstime / 1000 - minutes * 60)
			self.timeLabel.set_text("\n" + str(minutes) + " m " + str(seconds) + " s\n")
		else:
			# Check if there another song to play
			self.timeLabel.set_text("")
			file = self.playlistBrowser.getNextSong()
			if file != None:
				self.loadFile(self.currentdir+"/"+file)
		return True

	def loadFile(self, filename):
		self.file = filename
		self.currentdir = os.path.dirname(self.file)
		pygame.mixer.music.load(self.file)
		self.displayStatus("File: " + os.path.basename(self.file))
		self.paused = False
		self.play(None)

	"""
	def load(self, widget, data=None):
		# I simply hide the file chooser dialog around instead of creating and destroying it each time because on the Nokia Internet tablet it takes ages to load a large directory of midi files. Note that this method will not refresh the file list.
		if self.file != None:
			#print "setting "+self.file
			self.fileChooser.set_filename(self.file)
		response = self.fileChooser.run()
		if response == gtk.RESPONSE_OK:
			self.loadFile(self.fileChooser.get_filename())
		self.fileChooser.hide()
	"""

	def play(self, widget, data=None):
		if self.isPaused:
			pygame.mixer.music.unpause()
			self.isPaused = False
		else:
			# play only if mixer is not busy (i.e. must stop first)
			if not pygame.mixer.music.get_busy():
				pygame.mixer.music.play()
				self.isPaused = False

	def pause(self, widget, data=None):
		# Only need to pause if mixer is busy
		if pygame.mixer.music.get_busy():
			pygame.mixer.music.pause()
			self.isPaused = True
			

	def stop(self, widget, data=None):
		pygame.mixer.music.stop()
		self.isPaused = False

	def delete_event(self, widget, event, data=None):
		pygame.mixer.music.stop()
		pygame.quit()
		gobject.source_remove(self.source_id)
		gtk.main_quit()
		return False

	def __init__(self):
		pygame.init()
		#self.clock=pygame.time.Clock()
		if len(sys.argv)>=2 and sys.argv[1] != "":
			self.file=sys.argv[1]
			self.currentdir = os.path.realpath(os.path.dirname(sys.argv[1]))
			pygame.mixer.music.load(self.file)
			pygame.mixer.music.play()
		else:
			self.file=None
			self.currentdir = os.path.realpath(os.path.dirname(sys.argv[0]))
		print self.currentdir
		self.isPaused = False;

		pygame.mixer.init()
		pygame.mixer.music.set_volume(1.0)
		print pygame.mixer.get_init()
		print pygame.mixer.music.get_volume()

		if isHildon:
			self.app = hildon.Program()
			self.window = hildon.Window()
		else:
			self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
		self.window.set_title(APP_TITLE)
		self.window.connect("delete_event", self.delete_event)
		self.window.set_border_width(10)
		self.window.connect("key-press-event", self.on_key_press) 
		self.window.connect("window-state-event", self.on_window_state_change) 
		self.window_in_fullscreen = False #The window isn't in full screen mode initially.
		self.vbox = gtk.VBox(False, 0)

		self.hbox = gtk.HBox(False, 0)

		self.browserFrame = gtk.Frame("")
		self.browserFrame.show()
		self.playlistFrame = gtk.Frame("")
		self.playlistFrame.show()
		global FileBrowser, PlayListBrowser
		self.fileBrowser = FileBrowser(self.browserFrame, self)
		self.fileBrowser.refreshFiles()
		self.playlistBrowser = PlayListBrowser(self.playlistFrame, self)

		self.playBtn = gtk.Button("\nPlay\n")
		self.playBtn.connect("clicked", self.play, None)

		self.pauseBtn = gtk.Button("Pause")
		self.pauseBtn.connect("clicked", self.pause, None)
		
		self.stopBtn = gtk.Button("Stop")
		self.stopBtn.connect("clicked", self.stop, None)
		
		#self.loadBtn = gtk.Button("Load")
		#self.loadBtn.connect("clicked", self.fileBrowser.playSelected, None)
		
		self.hbox.pack_start(self.playBtn, True, True, 0)
		self.hbox.pack_start(self.pauseBtn, True, True, 0)
		self.hbox.pack_start(self.stopBtn, True, True, 0)
		#self.hbox.pack_start(self.loadBtn, True, True, 0)		

		self.hbox2 = gtk.HBox(False,0)
		self.statusLabel = gtk.Label("")
		if self.file == None:
			self.displayStatus("No file loaded")
		else:
			self.displayStatus("File: " + os.path.basename(self.file))
		self.hbox2.pack_start(self.statusLabel, False, False, 0)

		self.timeLabel = gtk.Label("")
		self.hbox2.pack_end(self.timeLabel, False, False, 0)

		# About
		self.aboutFrame = gtk.Frame()
		self.Label2 = gtk.Label(APP_ABOUT)
		self.Label2.set_line_wrap(True)
		self.aboutFrame.add(self.Label2)
		self.aboutFrame.show()

		self.notebook = gtk.Notebook()
		self.notebook.set_tab_pos(gtk.POS_TOP)
		
		self.notebook.append_page(self.browserFrame, gtk.Label("File browser"))
		self.notebook.append_page(self.playlistFrame, gtk.Label("Playlist"))
		self.notebook.append_page(self.aboutFrame, gtk.Label("About"))
		self.notebook.show()

		self.vbox.pack_start(self.hbox2, False, False, 0)
		self.vbox.pack_start(self.hbox, False, False, 0)
		self.vbox.pack_start(self.notebook, True, True, 0)

		self.window.add(self.vbox)
		self.hbox.show()
		self.playBtn.show()
		self.pauseBtn.show()
		self.stopBtn.show()
		#self.loadBtn.show()

		self.hbox2.show()
		self.statusLabel.show()
		self.timeLabel.show()

		self.Label2.show()

		self.vbox.show()
		self.window.show()

		self.source_id = gobject.timeout_add(250, self.doIdle)

		"""
		if isHildon:
			self.fileChooser = hildon.FileChooserDialog(self.window, gtk.FILE_CHOOSER_ACTION_OPEN)
		else:
			self.fileChooser = gtk.FileChooserDialog("Load file", self.window, gtk.FILE_CHOOSER_ACTION_OPEN, (gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL,gtk.STOCK_OPEN,gtk.RESPONSE_OK))
		self.fileChooser.set_default_response(gtk.RESPONSE_OK)
		filter = gtk.FileFilter()
		filter.set_name("MIDI files")
		filter.add_pattern("*.mid")
		self.fileChooser.add_filter(filter)
		self.fileChooser.set_current_folder(self.currentdir)
		"""

	def on_window_state_change(self, widget, event, *args): 
		if event.new_window_state & gtk.gdk.WINDOW_STATE_FULLSCREEN: 
			self.window_in_fullscreen = True 
		else: 
			self.window_in_fullscreen = False 

	def on_key_press(self, widget, event, *args):
		if event.keyval == gtk.keysyms.F6:
			# The "Full screen" hardware key has been pressed
			if self.window_in_fullscreen:
				self.window.unfullscreen ()
			else:
				self.window.fullscreen ()

	def main(self):
		gtk.main()

class PlayListBrowser:
	def __init__(self, window, app):
		self.app = app
		self.vbox = gtk.VBox(False, 0)

		self.playListWin = gtk.ScrolledWindow()
		self.playListWin.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
		self.playlist = gtk.ListStore(str)
		self.playListView = gtk.TreeView(self.playlist)
		self.playlistcol = gtk.TreeViewColumn('')
		self.playListView.append_column(self.playlistcol)
		self.playListView.set_headers_visible(False)
		self.cell = gtk.CellRendererText()
		self.playlistcol.pack_start(self.cell, True)
		self.playlistcol.add_attribute(self.cell, 'text', 0)
		self.playListWin.add(self.playListView)
		self.playListView.show()
		self.playListWin.show()

		self.hbox = gtk.HBox(False, 0)
		self.clearBtn = gtk.Button("Clear list")
		self.clearBtn.connect("clicked", self.clearList)
		self.clearBtn.show()
		self.hbox.pack_start(self.clearBtn, True, True, 0)
		self.delBtn = gtk.Button("Delete selected")
		self.delBtn.connect("clicked", self.delSelected)
		self.delBtn.show()
		self.hbox.pack_start(self.delBtn, True, True, 0)

		self.hbox.show()

		self.vbox.pack_start(self.playListWin, True, True, 0)
		self.vbox.pack_start(self.hbox, False, False, 0)
		window.add(self.vbox)
		self.vbox.show()
		window.show()

	def clearList(self, widget, data=None):
		self.playlist.clear()

	def delSelected(self, widget, data=None):
		selection = self.playListView.get_selection()
		(model, iter) = selection.get_selected()
		if iter != None:
			value = self.playlist.get_value(iter, 0)
			path = self.playlist.get_path(iter)
			self.playlist.remove(iter)

			#print "Removed " + value

	def getNextSong(self):
		iter = self.playlist.get_iter_first()
		if iter != None:
			value = self.playlist.get_value(iter, 0)
			path = self.playlist.get_path(iter)
			self.playlist.remove(iter)
			#print "Next song is " + value
			return value
		else:
			return None


class FileBrowser:

	def __init__(self, window, app):
		self.app = app
		self.vbox = gtk.VBox(False, 0)
		self.label = gtk.Label(self.app.currentdir)
		self.hbox = gtk.HBox(False, 0)

		self.vbox.pack_start(self.label, False, False, 0)
		self.vbox.pack_start(self.hbox, True, True, 0)

		self.label.show()

		self.dirlist = gtk.ListStore(str)
		self.dirListWin = gtk.ScrolledWindow()
		self.dirListWin.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_ALWAYS)
		self.dirListSorted = gtk.TreeModelSort(self.dirlist)
		self.dirListSorted.set_sort_column_id(0, gtk.SORT_ASCENDING)
		self.dirListView = gtk.TreeView(self.dirListSorted)
		self.dirlistcol = gtk.TreeViewColumn('')
		self.dirListView.append_column(self.dirlistcol)
		self.dirListView.set_headers_visible(False)
		self.cell1 = gtk.CellRendererText()
		self.dirlistcol.pack_start(self.cell1, True)
		self.dirlistcol.add_attribute(self.cell1, 'text', 0)
		self.dirListWin.add(self.dirListView)
		self.dirListView.show()
		self.dirListWin.show()
		self.dirListView.connect("row-activated", self.dirSelected)

		self.filelist = gtk.ListStore(str)
		self.fileListWin = gtk.ScrolledWindow()
		self.fileListWin.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_ALWAYS)
		self.fileListSorted = gtk.TreeModelSort(self.filelist)
		self.fileListSorted.set_sort_column_id(0, gtk.SORT_ASCENDING)
		self.fileListView = gtk.TreeView(self.fileListSorted)
		self.filelistcol = gtk.TreeViewColumn('')
		self.fileListView.append_column(self.filelistcol)
		self.fileListView.set_headers_visible(False)
		self.cell2 = gtk.CellRendererText()
		self.filelistcol.pack_start(self.cell2, True)
		self.filelistcol.add_attribute(self.cell2, 'text', 0)
		self.fileListWin.add(self.fileListView)
		self.fileListView.show()
		self.fileListWin.show()
		self.fileListView.connect("row-activated", self.playSelected)

		self.hbox.pack_start(self.dirListWin, True, True, 0)
		self.hbox.pack_start(self.fileListWin, True, True, 0)
		self.hbox.show()

		# Queue and load buttons
		self.hbox2 = gtk.HBox(False, 0)
		self.queueBtn = gtk.Button("Add to playlist")
		self.queueBtn.connect("clicked", self.queueSelected)
		self.queueBtn.show()
		self.loadBtn = gtk.Button("Play now")
		self.loadBtn.connect("clicked", self.playSelected)
		self.loadBtn.show()

		self.hbox2.pack_start(self.queueBtn, True, True, 0)
		self.hbox2.pack_start(self.loadBtn, True, True, 0)
		self.hbox2.show()
		self.vbox.pack_start(self.hbox2, False, False, 0)

		window.add(self.vbox)
		self.vbox.show()
		window.show()

	def refreshFiles(self):
		# Clear the lists
		self.dirlist.clear()
		self.filelist.clear()
		if self.app.currentdir != "":
			self.dirlist.append([".."])

		if self.app.currentdir != "":
			temp = os.listdir(self.app.currentdir)
			self.label.set_text(self.app.currentdir)
		else:
			temp = os.listdir("/")
			self.label.set_text("/")
		for file in temp:
			if os.path.isdir(self.app.currentdir +"/"+ file):
				self.dirlist.append([file])
			elif file.endswith("mid"):
				self.filelist.append([file])

	def queueSelected(self, widget, event=None, data=None):
		selection = self.fileListView.get_selection()
		(model, iter) = selection.get_selected()
		value = self.fileListSorted.get_value(iter, 0)
		#print value + " selected!"

		# queue the file
		self.app.playlistBrowser.playlist.append([value])

	def playSelected(self, widget, event=None, data=None):
		selection = self.fileListView.get_selection()
		(model, iter) = selection.get_selected()
		if iter != None:
			value = self.fileListSorted.get_value(iter, 0)
			#print value + " selected!"

			# play the file
			self.app.loadFile(self.app.currentdir+"/"+value)
	
	def dirSelected(self, widget, event, data=None):
		selection = self.dirListView.get_selection()
		(model, iter) = selection.get_selected()
		if iter != None:
			value = self.dirListSorted.get_value(iter, 0)
			if value == "..": # go up
				pos = self.app.currentdir.rfind("/")
				if pos != -1:
					value = self.app.currentdir[0:pos]
				else:
					value = self.app.currentdir
			else:
				value = self.app.currentdir + "/" + value
			#print "Switching to folder " + value
			self.app.currentdir = value
			self.refreshFiles()


if __name__ == "__main__":
	app = PlayMus()
	app.main()



