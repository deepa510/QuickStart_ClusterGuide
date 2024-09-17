# QuickStart Guide for Connecting to SLURM Cluster

**Objective**: This document provides a stepwise guide for connecting to SLURM.

> **Note**: In order to connect, please make sure to connect to **USF VPN**.

---

## Connecting via SSH

The following information will be needed to connect via SSH:

- **Your USF NetID and Password**
- **Hostname**: `circe.rc.usf.edu`
- **SSH Port**: 22 (default)

### SSH Clients for Windows

- **PuTTY**: [PuTTY Download Link](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)  
  *Note*: PuTTY is the recommended client to use when connecting to CIRCE. IT staff will provide full support for users utilizing this connection method. However, graphical (X11) connections are not provided.
  
- **Cygwin (Includes OpenSSH)**: [Cygwin Download Link](http://www.cygwin.com/)  
  *Note*: Cygwin is for advanced users who are familiar with using a UNIX/Linux environment! We can only provide limited support for this method, so be warned.

### SSH Clients for Mac OSX and Linux

- **OSX SSH Tutorial**: [OSX SSH Guide](http://osxdaily.com/2017/04/28/howto-ssh-client-mac/)
- **Linux SSH Tutorial**: [Linux SSH Guide](https://acloudguru.com/blog/engineering/ssh-and-scp-howto-tips-tricks)

---

## Connecting via X2Go

For **Windows**, **Mac**, and **Linux**, you can also use **X2Go**. X2Go enables you to access a graphical desktop of a computer over a low bandwidth (or high bandwidth) connection. This method uses the **MATE** desktop environment to provide access to CIRCE and its applications.

### Installing the X2Go Client

Please click the appropriate link for X2Go for your operating system:

- **Windows (v4.1.2.0)**: [Windows Download Link](https://code.x2go.org/releases/binary-win32/x2goclient/releases/4.1.2.0-2018.06.22/x2goclient-4.1.2.0-2018.06.22-setup.exe)  
  *Note*: For Windows, you must use v4.1.2.0 to connect to CIRCE!

- **Mac OSX 10.11 and higher**: [Mac OSX 10.11 Download Link](http://code.x2go.org/releases/X2GoClient_latest_macosx_10_11.dmg)
- **Mac OSX 10.13 and higher**: [Mac OSX 10.13 Download Link](http://code.x2go.org/releases/X2GoClient_latest_macosx_10_13.dmg)
- **Ubuntu/Debian**: [Ubuntu/Debian Installation Guide](https://wiki.x2go.org/doku.php/doc:installation:x2goclient#ubuntu_debian)
- **Red Hat**: [Red Hat Installation Guide](https://wiki.x2go.org/doku.php/doc:installation:x2goclient#redhat)
- **Fedora**: [Fedora Installation Guide](https://wiki.x2go.org/doku.php/doc:installation:x2goclient#fedora)

For all operating systems, the directions for installation are available from the vendor website.

---

## Connecting to CIRCE via X2Go

### Setting up a Graphical Desktop Session

### Configuring X2Go

When you first log into CIRCE via X2Go, you will begin a new session. A session provides a full desktop environment and will include any applications you launch while you are logged in. This will not include any programs running in the queue.

### Session Settings

1. **Enter the host**: `circe.rc.usf.edu`
2. **Enter your USF NetID** next to "Login"
3. **Leave the SSH Port** as 22
4. **Set the Session type** to **MATE** from the drop-down list.

Use the below image as a guide:  
![X2Go Session Settings](https://github.com/user-attachments/assets/369cf594-1426-462e-b40d-ae6e28bc5d56)

---

### Connection Settings

- You can tailor the connection speed to fit your network type, but the default ADSL is fine for most connections and may eliminate some latency issues.
- The **4k-png compression method** has been the most stable in testing, while providing reasonable resolution.

> *Tinker at your own risk!*

![Connection Settings](https://github.com/user-attachments/assets/91761772-5107-4a7f-9dee-ecd827e224ee)

---

### Input/Output Settings

For Display settings:

- Select **Custom** and start with size **1024x768**.
- Additionally, the display can be adjusted to “Fullscreen” or “Use whole display” for multiple monitor setups.  
- Uncheck the box for **Set display DPI**. Leaving this active may cause some applications to display improperly.

![Input/Output Settings](https://github.com/user-attachments/assets/24d67b0f-e4ca-422b-be88-5df854a39252)

---

### Media Settings

- **Sound and printing** can be turned off to improve speed and stability.  
- Most other settings can be left as the default.

![Media Settings](https://github.com/user-attachments/assets/660f1809-f773-45d7-aa34-51b7a4a93fbd)

---

### Shared Folders Settings

- The "Shared Folders" tab can remain as the default, shown below:

![Shared Folders Settings](https://github.com/user-attachments/assets/51fc1658-793c-4a00-af9a-cbb636c17479)

---

## Logging In

- Select the appropriate session box on the right side of the X2Go window representing your CIRCE session.
- You should be prompted for your **NetID password**.
- Upon selecting **OK**, a new connection will be initiated. It may take several seconds for the desktop to initialize, so please be patient.

> **Mac users**: You may need to open the desktop settings once X2Go is running and disable all keyboard shortcuts on the Linux system. There seems to be a bug that may cause X2Go to think the "Super Key" is continually pressed.

---

## Minimizing from Fullscreen Mode

Active X2Go windows can be minimized to the background without closing the session. The method varies by OS:

- **Windows**: Use **Alt+Tab** to switch windows and place X2Go in the background.
- **Linux**: Use **Ctrl+Alt+m** to minimize X2Go to the panel.
- **Mac**: Use the top Mac menu.

---

## Ending a Session

Sessions can be ended by logging out (going to **System -> Logoff** in the top toolbar).

> **NOTE**: Please do not click the **X** in the upper-right-hand corner to exit, as this will suspend your session and not exit.

---

## Suspending Sessions

- Session suspension and resume is **not fully supported**.  
- All new connections may or may not restore previous sessions regardless of whether you attempted to suspend an existing session currently open.  
- Plan accordingly.
